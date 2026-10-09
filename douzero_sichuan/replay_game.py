"""Observe a frozen evaluation game and export its complete human-readable replay."""
import argparse
from collections import Counter
from contextlib import ExitStack
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

from game import MahjongGame
from . import core

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / 'DouZero_Sichuan_Formal_20260926'
NAMES = {'start_game': '发牌', '_execute_exchange': '换三张完成',
         'select_exchange_tiles': '提交换三张', 'set_missing_suit': '定缺',
         'draw_tile': '摸牌', 'discard_tile': '弃牌', 'peng': '碰',
         'gang': '杠', 'hu': '胡', 'check_game_over': '终局'}


def snapshot(game):
    players = []
    for p in game.players:
        players.append(dict(pid=p.player_id, hand=[str(t) for t in p.hand_tiles],
                            melds=[[str(t) for t in m] for m in p.open_melds],
                            river=[str(t) for t in p.discarded_tiles],
                            missing=p.missing_suit.value if p.missing_suit else None,
                            balance=p.balance, hu=p.is_hu,
                            win_tile=str(p.hu_win_tile) if p.hu_win_tile else None,
                            self_draw=p.hu_is_self_drawn))
    return dict(players=players, wall=game.deck.remaining_count(),
                current=game.current_player_id, over=game.is_game_over,
                winners=list(game.winners), phase=game.phase.value)


def conservation(game):
    counts = Counter(core.tile_index(t) for p in game.players for t in
                     p.hand_tiles + p.discarded_tiles + [t for m in p.open_melds for t in m])
    counts.update(core.tile_index(t) for t in game.deck.tiles)
    counts.update(core.tile_index(p.hu_win_tile) for p in game.players
                  if p.is_hu and not p.hu_is_self_drawn and p.hu_win_tile is not None)
    return len(counts) == 27 and all(n == 4 for n in counts.values())


@lru_cache(maxsize=4096)
def decompose(counts):
    if not any(counts):
        return ()
    i = next(i for i, n in enumerate(counts) if n)
    c = list(counts)
    if c[i] >= 3:
        c[i] -= 3
        rest = decompose(tuple(c))
        if rest is not None:
            return ((i, i, i),) + rest
        c[i] += 3
    if i % 9 <= 6 and c[i + 1] and c[i + 2]:
        for j in (i, i + 1, i + 2):
            c[j] -= 1
        rest = decompose(tuple(c))
        if rest is not None:
            return ((i, i + 1, i + 2),) + rest
    return None


def independent_hu(game, pid, tile, self_draw):
    p = game.players[pid]
    tiles = list(p.hand_tiles) + ([] if self_draw else [tile])
    all_tiles = tiles + [t for m in p.open_melds for t in m]
    if any(t.suit == p.missing_suit for t in all_tiles):
        return dict(valid=False, reason='仍有定缺花色')
    if any(n > 4 for n in Counter(core.tile_index(t) for t in all_tiles).values()):
        return dict(valid=False, reason='同牌超过四张')
    if any(len(m) not in (3, 4) or len(set(str(t) for t in m)) != 1 for m in p.open_melds):
        return dict(valid=False, reason='固定副露不合法')
    counts = [0] * 27
    for t in tiles:
        counts[core.tile_index(t)] += 1
    labels = [f'{n}{s.value}' for s in core.SUITS for n in range(1, 10)]
    if not p.open_melds and len(tiles) == 14 and all(n % 2 == 0 for n in counts):
        return dict(valid=True, pattern='七对', groups=[[labels[i]] * 2
                    for i, n in enumerate(counts) for _ in range(n // 2)])
    if len(tiles) != 3 * (4 - len(p.open_melds)) + 2:
        return dict(valid=False, reason='暗手张数与副露数量不符')
    for i, n in enumerate(counts):
        if n >= 2:
            c = counts.copy()
            c[i] -= 2
            groups = decompose(tuple(c))
            if groups is not None:
                return dict(valid=True, pattern='标准胡', pair=[labels[i]] * 2,
                            groups=[[labels[j] for j in g] for g in groups],
                            fixed_melds=[[str(t) for t in m] for m in p.open_melds])
    return dict(valid=False, reason='不能拆成固定副露、面子和将牌')


class Observer:
    def __init__(self):
        self.events = []
        self.turn = 0
        self.encoding = 0
        self.active = False

    def record(self, game, kind, text, **details):
        turn = self.turn + 1 if kind == 'draw_tile' else self.turn
        self.events.append(dict(index=len(self.events) + 1, turn=turn,
                                kind=kind, text=text, state=snapshot(game), **details))

    def wrapper(self, name, original):
        def call(game, *args, **kwargs):
            before = snapshot(game)
            ledger_start = len(game.settlement_ledger)
            hu_check = independent_hu(game, args[0], args[1],
                                     args[2] if len(args) > 2 else kwargs.get('is_self_drawn', True)) if name == 'hu' else None
            result = original(game, *args, **kwargs)
            if name == 'check_responses':
                if result:
                    self.record(game, 'offers', f'对 P{args[1]} 的 {args[0]}：' +
                                '；'.join(f'P{pid} 可 {", ".join(acts)}' for pid, acts in result.items()),
                                offers=result)
                return result
            if name == 'check_game_over' and (not result or before['over']):
                return result
            if name in ('discard_tile', 'peng', 'select_exchange_tiles', 'set_missing_suit') and not result:
                self.record(game, 'rejected', f'{NAMES[name]}被拒绝', arguments=[str(a) for a in args])
                return result
            text = NAMES[name]
            if args:
                text = f'P{args[0]} {text}'
            if name in ('discard_tile', 'peng', 'gang', 'hu'):
                text += f' {args[1]}'
            if name == 'draw_tile':
                text += f' {result}'
            if name == 'select_exchange_tiles':
                text += ' ' + ' '.join(str(t) for t in args[1])
            if name == 'set_missing_suit':
                text += ' ' + args[1].value
            details = dict(before=before, tile_counts_valid=conservation(game),
                           transfers=[dict(e) for e in game.settlement_ledger[ledger_start:]])
            if name == '_execute_exchange':
                text += f'（{game.exchange_direction.value}；庄家补一张）'
                details['exchanges'] = [dict(pid=p.player_id, sent=[str(t) for t in p.exchange_tiles],
                                            received=[str(t) for t in p.received_tiles]) for p in game.players]
            if name == 'peng':
                pid, tile, source = args
                text += f'（碰 P{source}）'
                after = snapshot(game)
                details['peng_valid'] = (
                    Counter(before['players'][pid]['hand'])[str(tile)] -
                    Counter(after['players'][pid]['hand'])[str(tile)] == 2 and
                    len(after['players'][pid]['melds']) == len(before['players'][pid]['melds']) + 1 and
                    after['players'][pid]['melds'][-1] == [str(tile)] * 3 and
                    before['players'][source]['river'][-1:] == [str(tile)] and
                    after['players'][source]['river'] == before['players'][source]['river'][:-1])
            if name == 'gang':
                text += f'（{game.history[-1]["desc"]}；补摸 {result[1]}）'
                details['replacement_tile'] = str(result[1])
                details['accepted'] = result[0]
            if name == 'hu':
                self_draw = args[2] if len(args) > 2 else kwargs.get('is_self_drawn', True)
                source = args[3] if len(args) > 3 else kwargs.get('discard_player_id')
                text += '（自摸）' if self_draw else f'（荣和 P{source}）'
                text += f'：{result[0]}番，' + ' / '.join(result[1])
                details.update(hu_check=hu_check, fan=result[0], fan_types=result[1])
            self.record(game, name, text, **details)
            return result
        return call

    def replay(self, model, row):
        original_encode, original_tile = core.encode, core.action_tile
        original_init = core.init_game

        def encode(*args, **kwargs):
            self.encoding += 1
            try:
                return original_encode(*args, **kwargs)
            finally:
                self.encoding -= 1

        def action_tile(game, pid, action, response):
            tile = original_tile(game, pid, action, response)
            if self.active and not self.encoding:
                if not response:
                    self.turn += 1
                labels = {'h': '选择胡', 'p': '选择碰', 'g': '选择杠', 'n': '过'}
                label = labels.get(action, f'选择弃牌 {tile}')
                self.record(game, 'decision', f'P{pid} {label}' + ('（响应弃牌）' if response else ''),
                            pid=pid, action=action, response=response)
            return tile

        def init(*args, **kwargs):
            result = original_init(*args, **kwargs)
            self.active = True
            self.record(result[0], 'ready', '换牌、定缺完成；开始打牌')
            return result

        with ExitStack() as stack:
            for name in (*NAMES, 'check_responses'):
                stack.enter_context(patch.object(MahjongGame, name, self.wrapper(name, getattr(MahjongGame, name))))
            stack.enter_context(patch.object(core, 'encode', encode))
            stack.enter_context(patch.object(core, 'action_tile', action_tile))
            stack.enter_context(patch.object(core, 'init_game', init))
            metrics = core.episode(model, row['seed'], 0, True, 'holdout')[3]
        if metrics != row['four_seat_metrics']:
            raise RuntimeError(f'Replay differs from recorded metrics: {metrics}')
        return metrics


def render_markdown(report):
    lines = ['# DouZero-adapted 完整对局重放', '',
             f'随机种子：`{report["seed"]}`。P0 为冻结 DouZero，P1–P3 为原评测的课程 holdout 对手。', '',
             '以下四家手牌是审计展示；原策略仍只接收原本允许的观测。行动轮次按实际决策计数，碰后出牌和杠后出牌会改变正常座次顺序。', '',
             f'重放结果与原正式记录完全一致：`{report["metrics"]["net"]}`；四家净分和为零。', '',
             '## 关键动作', '', '| 行动轮次 | 动作 |', '|---:|---|']
    for e in report['events']:
        if e['kind'] in ('peng', 'gang', 'hu', 'check_game_over'):
            lines.append(f'| {e["turn"]} | {e["text"]} |')
    lines += ['', '## 完整动作序列', '', '| 事件 | 行动轮次 | 动作 / 响应 | 牌墙剩余 |', '|---:|---:|---|---:|']
    for e in report['events']:
        lines.append(f'| {e["index"]} | {e["turn"]} | {e["text"]} | {e["state"]["wall"]} |')
    lines += ['', '## 每个动作后的四家手牌', '']
    for e in report['events']:
        lines += [f'### 事件 {e["index"]}：{e["text"]}', '',
                  '| 玩家 | 定缺 | 暗手 | 固定副露 | 牌河 | 胡牌张 | 当前余额 |',
                  '|---|---|---|---|---|---|---:|']
        for p in e['state']['players']:
            lines.append(f'| P{p["pid"]} | {p["missing"] or "未定"} | {" ".join(p["hand"])} | '
                         f'{" / ".join(" ".join(m) for m in p["melds"]) or "无"} | '
                         f'{" ".join(p["river"]) or "无"} | {p["win_tile"] or "无"} | {p["balance"]} |')
        if 'hu_check' in e:
            lines += ['', '独立胡型拆分：`' + json.dumps(e['hu_check'], ensure_ascii=False) + '`。']
        if 'peng_valid' in e:
            lines += ['', f'碰牌数量与牌河转移核对：{"通过" if e["peng_valid"] else "失败"}。']
        for t in e.get('transfers', []):
            lines += ['', f'结算：P{t["payer_id"]} → P{t["receiver_id"]}，{t["score"]} 分，`{t["kind"]}`。']
        lines.append('')
    lines += ['## 审计范围与已知问题', '',
              f'本局核对：`{json.dumps(report["checks"], ensure_ascii=False)}`。', '',
              '独立胡型核对检查面子、将牌、固定副露、定缺和四张上限，不等于独立验证番型与全部四川麻将地方规则。', '',
              '已有规则边界测试记录（不是声称都在本局发生）：', '',
              '- 终局检查重复调用可能重复查叫结算。',
              '- 带幺九判断可能把已固定的碰牌重新拆成顺子。',
              '- 首张弃牌标记提前清除，影响地胡判断。',
              '- 杠牌未立即转移杠分；是否保留此简化需要明确规则版本。', '',
              '边界测试来源：`../DouZero_Sichuan_Formal_20260926/settlement_audit/rule_boundary_fixtures.json`。', '',
              '此外，历史 DouZero 执行器按玩家编号处理响应，遇到第一个接受的碰、杠或胡就停止。该逻辑未统一实现胡优先及一炮多响；本局没有触发碰与其他玩家胡竞争的情况，不能用本局证明响应优先级正确。', '',
              '本重放不改变历史评测规则、模型或结果；完整重放不能证明引擎不存在其他问题。', '']
    return '\n'.join(lines)


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>麻将完整对局重放</title><style>
*{box-sizing:border-box}body{margin:0;font:15px/1.6 system-ui,sans-serif;color:#20282a;background:#f6f7f8}
header,main{max-width:1300px;margin:auto;padding:16px}h1{font-size:24px;margin:0}p{margin:8px 0}
.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:12px 0;border-bottom:1px solid #c9cdd0}
button{width:40px;height:36px;border:1px solid #aab2b5;background:white;border-radius:4px;font-size:20px;cursor:pointer}select{height:36px;max-width:100%}
input{flex:1;min-width:140px}#title{font-size:18px;font-weight:600;padding:12px 0;color:#155f52}
.players{display:grid;grid-template-columns:1fr 1fr;gap:12px}.player{background:white;border:1px solid #cbd0d3;padding:12px;border-radius:4px;min-width:0}
.player h2{font-size:17px;margin:0 0 8px}.tiles{display:flex;gap:4px;flex-wrap:wrap;min-height:36px}
.tile{display:inline-flex;justify-content:center;align-items:center;width:32px;height:40px;background:#fff;border:1px solid #a6adaf;border-bottom:3px solid #a6adaf;border-radius:3px;font-weight:600}
.wan{color:#b62d3e}.tiao{color:#19734a}.tong{color:#295ba6}.label{font-size:13px;color:#616e73;margin:8px 0 3px}.meld{display:inline-flex;gap:2px;margin-right:12px}.river .tile{height:30px}
#details{padding:12px 0;white-space:pre-wrap;overflow-wrap:anywhere}details{border-top:1px solid #c9cdd0;padding-top:12px}table{width:100%;border-collapse:collapse}td,th{padding:7px;text-align:left;border-bottom:1px solid #ddd}tr[data-index]{cursor:pointer}tr.active{background:#d7ede7}
@media(max-width:650px){.players{grid-template-columns:1fr}header,main{padding:12px}.tile{width:29px}h1{font-size:20px}}
</style><header><h1>DouZero-adapted 完整对局</h1><p id="meta"></p><p>四家手牌仅供规则审计，策略运行时的观测权限保持原样。</p></header>
<main><div class="toolbar"><button id="prev" title="上一步" aria-label="上一步">←</button><button id="next" title="下一步" aria-label="下一步">→</button><input id="slider" type="range" min="0" value="0" aria-label="事件进度"><span id="position"></span><select id="jump" aria-label="关键动作"><option value="0">关键动作</option></select></div>
<div id="title"></div><div class="players" id="players"></div><div id="details"></div>
<details><summary>完整动作序列</summary><table><thead><tr><th>事件</th><th>行动轮次</th><th>动作</th><th>牌墙</th></tr></thead><tbody id="log"></tbody></table></details>
<details><summary>核对结果与已知规则边界</summary><p id="checks"></p><p>独立胡型核对检查面子、将牌、固定副露和定缺；未独立验证全部番型。已有边界测试发现重复终局结算、带幺九重排固定副露、地胡标记，以及没有即时杠分的问题。这些不等于都在本局发生。历史 DouZero 执行器也未统一处理胡优先与一炮多响，本局不能验证这类竞争响应规则。</p></details></main>
<script id="data" type="application/json">__DATA__</script><script>
const report=JSON.parse(document.getElementById('data').textContent),events=report.events;let index=0;
const $=id=>document.getElementById(id);function node(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n}
function tiles(list){const box=node('div',undefined,'tiles');list.forEach(t=>box.append(node('span',t,'tile '+(t.includes('万')?'wan':t.includes('条')?'tiao':'tong'))));if(!list.length)box.textContent='无';return box}
$('meta').textContent='Seed '+report.seed+' · 原记录净分 '+report.metrics.net.join(' / ')+' · 重放完全一致';$('slider').max=events.length-1;
events.forEach((e,i)=>{const tr=node('tr');tr.dataset.index=i;[e.index,e.turn,e.text,e.state.wall].forEach(x=>tr.append(node('td',x)));tr.onclick=()=>show(i);$('log').append(tr);if(['ready','peng','gang','hu','check_game_over'].includes(e.kind)){const option=node('option',e.text);option.value=i;$('jump').append(option)}});
function show(i){index=Math.max(0,Math.min(i,events.length-1));const e=events[index];$('slider').value=index;$('position').textContent=(index+1)+' / '+events.length;$('title').textContent='行动轮次 '+e.turn+' · '+e.text+' · 牌墙 '+e.state.wall;$('players').replaceChildren();
e.state.players.forEach(p=>{const box=node('section',undefined,'player');box.append(node('h2','P'+p.pid+(p.pid===0?' · DouZero':' · 课程对手')+' · 缺'+(p.missing||'未定')+' · 余额 '+p.balance+(p.hu?' · 已胡 '+p.win_tile:'')));box.append(node('div','暗手','label'),tiles(p.hand),node('div','固定副露','label'));const melds=node('div');p.melds.forEach(m=>{const b=tiles(m);b.className='meld';melds.append(b)});if(!p.melds.length)melds.textContent='无';box.append(melds,node('div','牌河','label'));const river=tiles(p.river);river.classList.add('river');box.append(river);$('players').append(box)});
const parts=[];if(e.hu_check){const c=e.hu_check;parts.push('独立胡型核对：'+(c.valid?'通过 · '+c.pattern:'失败 · '+c.reason));if(c.pair)parts.push('将牌：'+c.pair.join(' '));if(c.groups)parts.push('暗手面子 / 对子：'+c.groups.map(g=>g.join(' ')).join(' / '));if(c.fixed_melds&&c.fixed_melds.length)parts.push('固定副露：'+c.fixed_melds.map(g=>g.join(' ')).join(' / '));}if('peng_valid'in e)parts.push('碰牌数量 / 牌河转移：'+(e.peng_valid?'通过':'失败'));(e.transfers||[]).forEach(t=>parts.push('P'+t.payer_id+' → P'+t.receiver_id+'：'+t.score+' 分（'+(t.kind==='hu_discard'?'荣和':t.kind==='hu_self_draw'?'自摸':t.kind)+'）'));$('details').textContent=parts.join('\n');document.querySelectorAll('tr[data-index]').forEach(tr=>tr.classList.toggle('active',Number(tr.dataset.index)===index));$('prev').disabled=index===0;$('next').disabled=index===events.length-1;}
const c=report.checks;$('checks').textContent='原记录与无观测重放一致：'+(c.original_metrics_equal&&c.uninstrumented_metrics_equal?'通过':'失败')+'；牌数守恒异常 '+c.tile_conservation_failures.length+' 次；'+c.peng_events+' 次碰，数量或碰后行动异常 '+(c.peng_failures.length+c.peng_followup_failures.length)+' 次；'+c.hu_events+' 次胡，独立胡型异常 '+c.independent_hu_failures.length+' 次；本局杠 '+c.gang_events+' 次。';$('prev').onclick=()=>show(index-1);$('next').onclick=()=>show(index+1);$('slider').oninput=e=>show(Number(e.target.value));$('jump').onchange=e=>show(Number(e.target.value));document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')show(index+1);if(e.key==='ArrowLeft')show(index-1)});show(0);
</script></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2026082001000000)
    parser.add_argument('--output', type=Path, default=ROOT / 'Mahjong_Replay_Audit_20261002')
    args = parser.parse_args()
    row = None
    for path in sorted((FORMAL / 'holdout').glob('seed_*/games.jsonl')):
        for line in path.read_text().splitlines():
            candidate = json.loads(line)
            if candidate['seed'] == args.seed:
                row = candidate
                break
        if row:
            break
    if row is None:
        raise ValueError('Seed not in formal evaluation')
    protocol = json.loads((FORMAL / 'PROTOCOL.json').read_text())
    hashes = {}
    for filename in ('game.py', 'run_gate1_experiments.py', 'prompt_builder.py',
                     'douzero_sichuan/core.py', 'Opponent_Curriculum_formal/screened.json'):
        actual = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
        expected = protocol['source_hashes'][str(ROOT / filename)]
        if actual != expected:
            raise ValueError(f'Frozen source changed: {filename}')
        hashes[filename] = actual
    torch.set_num_threads(1)
    model = core.ValueNetwork().eval()
    model.load_state_dict(torch.load(FORMAL / 'selected.pt', map_location='cpu', weights_only=True))
    observer = Observer()
    metrics = observer.replay(model, row)
    # Compare an uninstrumented run as well, so the observer cannot silently change play.
    plain_metrics = core.episode(model, args.seed, 0, True, 'holdout')[3]
    assert plain_metrics == metrics
    followup_failures = []
    for i, e in enumerate(observer.events):
        if e['kind'] != 'peng':
            continue
        pid = next(p['pid'] for p, b in zip(e['state']['players'], e['before']['players'])
                   if len(p['melds']) > len(b['melds']))
        subsequent = observer.events[i + 1:]
        next_decision = next((x for x in subsequent if x['kind'] == 'decision' and not x['response']), None)
        if next_decision is None or next_decision['pid'] != pid or any(
                x['kind'] == 'draw_tile' for x in subsequent if x['index'] < next_decision['index']):
            followup_failures.append(e['index'])
    checks = dict(original_metrics_equal=True, uninstrumented_metrics_equal=True,
                  tile_conservation_failures=[e['index'] for e in observer.events if e.get('tile_counts_valid') is False],
                  peng_events=sum(e['kind'] == 'peng' for e in observer.events),
                  peng_failures=[e['index'] for e in observer.events if e.get('peng_valid') is False],
                  peng_followup_failures=followup_failures,
                  gang_events=sum(e['kind'] == 'gang' for e in observer.events),
                  hu_events=sum(e['kind'] == 'hu' for e in observer.events),
                  independent_hu_failures=[e['index'] for e in observer.events if e.get('hu_check', {}).get('valid') is False])
    report = dict(seed=args.seed, method='DouZero-adapted', original_record=row,
                  checkpoint=str(FORMAL / 'selected.pt'),
                  checkpoint_sha256=hashlib.sha256((FORMAL / 'selected.pt').read_bytes()).hexdigest(),
                  source_hashes=hashes, metrics=metrics, checks=checks, events=observer.events)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / f'douzero_seed_{args.seed}'
    data = json.dumps(report, ensure_ascii=False, indent=2)
    stem.with_suffix('.json').write_text(data)
    stem.with_suffix('.md').write_text(render_markdown(report))
    stem.with_suffix('.html').write_text(HTML_TEMPLATE.replace('__DATA__', data.replace('<', '\\u003c')))
    print(json.dumps(dict(seed=args.seed, checks=checks, metrics=metrics, events=len(observer.events),
                         files=[str(stem.with_suffix(ext)) for ext in ('.html', '.md', '.json')]), ensure_ascii=False))


if __name__ == '__main__':
    main()
