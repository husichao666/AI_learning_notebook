#!/usr/bin/env python3
"""Render PP teaching timelines from public scheduler task orders.

Run: python3 scripts/generate_pp_timelines.py
Input JSON records the source URLs and SHA-256 of the inspected source files.
No framework, GPU, downloaded code, or network access is needed to regenerate.
The earliest-start compute model ignores communication. Zero Bubble diagrams
append illustrative local optimizer steps and the next iteration's prefix.
DualPipe pairs also use an illustrative duration, not a hardware measurement.
Every task is checked against model dependencies and physical-rank exclusivity.
"""
import json
import math
from collections import deque
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'scripts/data/pp_schedule_orders.json'
ASSETS = ROOT / 'docs/training/parallelism/assets'
TITLES = {
    'naive': ('朴素串行调度', 'P = 4 · M = 1'),
    'gpipe': ('GPipe', 'P = 4 · M = 6'),
    'onef1b': ('1F1B', 'P = 4 · M = 6'),
    'vpp': ('Interleaved 1F1B', 'P = 4 · C = 2 · M = 8'),
    'zb': ('Zero Bubble · ZB1P', 'P = 4 · M = 6'),
    'zb2p': ('Zero Bubble · ZB2P', 'P = 4 · M = 8'),
    'zbv': ('ZBV', 'P = 4 · C = 2 · M = 8'),
    'dualpipe': ('DualPipe', 'P = 4 · 每个方向 M = 6'),
    'dualpipev': ('DualPipeV', 'P = 4 · C = 2 · M = 8'),
}
COLORS = {'F': ('#dfe7ff', '#3f56a4'), 'B': ('#d9eee3', '#247249'),
          'X': ('#c9e9f3', '#236d83'), 'W': ('#ffebc4', '#956a20'),
          'O': ('#f5d6ea', '#963b78')}


def basic_schedule(name, p=4):
    """Build the introductory schedules with F=1 and full backward B=2."""
    m = 1 if name == 'naive' else 6
    orders = []
    for rank in range(p):
        forwards = [['F', rank, micro] for micro in range(m)]
        backwards = [['B', rank, micro] for micro in range(m)]
        if name == 'onef1b':
            warmup = min(p - rank - 1, m)
            local = forwards[:warmup]
            for i in range(m - warmup):
                local.extend([forwards[warmup + i], backwards[i]])
            local.extend(backwards[m - warmup:])
        else:
            local = forwards + list(reversed(backwards))
        orders.append([[op] for op in local])
    return dict(microbatches=m, paths=[list(range(p))], placement=list(range(p)),
                orders=orders, costs={'F': 1, 'B': 2})


def schedule(data):
    assert data['costs']['F'] == 1
    assert data['costs'].get('B', 2) == 2
    assert data['costs'].get('X', 1) == data['costs'].get('W', 1) == 1
    nodes, owner, rows = [], {}, []
    for rank, groups in enumerate(data['orders']):
        row = []
        for group in groups:
            index = len(nodes)
            ops = [tuple(x) for x in group]
            assert len(ops) in (1, 2)
            if len(ops) == 2:
                assert sorted(x[0] for x in ops) == ['B', 'F']
            node = dict(rank=rank, ops=ops, deps=set(), duration=data['costs']['pair']
                        if len(ops) == 2 else data['costs'][ops[0][0]])
            if row:
                node['deps'].add(row[-1])
            for op in ops:
                assert op not in owner, f'Duplicate operation: {op}'
                assert data['placement'][op[1]] == rank
                owner[op] = index
            row.append(index)
            nodes.append(node)
        rows.append(row)
    prev_stage, next_stage = {}, {}
    for path in data['paths']:
        for i, stage in enumerate(path):
            prev_stage[stage] = path[i - 1] if i else None
            next_stage[stage] = path[i + 1] if i + 1 < len(path) else None
    def backward(stage, micro):
        found = [op for op in [('X', stage, micro), ('B', stage, micro)] if op in owner]
        assert len(found) == 1, (stage, micro, found)
        return found[0]
    edges = []
    for stage in range(len(data['placement'])):
        for micro in range(data['microbatches']):
            f = ('F', stage, micro)
            assert f in owner
            b = backward(stage, micro)
            if b[0] == 'X':
                w = ('W', stage, micro)
                assert w in owner
                edges.append((b, w))
            else:
                assert ('W', stage, micro) not in owner
            edges.append((f, b))
            if prev_stage[stage] is not None:
                edges.append((('F', prev_stage[stage], micro), f))
            if next_stage[stage] is not None:
                edges.append((backward(next_stage[stage], micro), b))
    for before, after in edges:
        assert owner[before] != owner[after], 'An overlap pair contains dependent operations'
        nodes[owner[after]]['deps'].add(owner[before])
    followers = [[] for _ in nodes]
    degree = [len(n['deps']) for n in nodes]
    for i, n in enumerate(nodes):
        for dep in n['deps']:
            followers[dep].append(i)
    ready = deque(i for i, deg in enumerate(degree) if deg == 0)
    completed = 0
    while ready:
        i = ready.popleft()
        n = nodes[i]
        n['start'] = max((nodes[d]['end'] for d in n['deps']), default=0)
        n['end'] = n['start'] + n['duration']
        completed += 1
        for child in followers[i]:
            degree[child] -= 1
            if degree[child] == 0:
                ready.append(child)
    assert completed == len(nodes), 'Cyclic scheduling / unsatisfied dependencies'
    for before, after in edges:
        assert nodes[owner[before]]['end'] <= nodes[owner[after]]['start']
    for row in rows:
        for a, b in zip(row, row[1:]):
            assert nodes[a]['end'] <= nodes[b]['start'], 'Rank is double-booked'
        if 'activation_budget' in data:
            # Conservative teaching model: one chunk state stays live from F
            # through W; X releases nothing. This is not a byte-level estimate.
            live = 0
            for i in row:
                assert len(nodes[i]['ops']) == 1
                live += {'F': 1, 'X': 0, 'W': -1}[nodes[i]['ops'][0][0]]
                assert 0 <= live <= data['activation_budget'], 'Activation budget exceeded'
            assert live == 0, 'Activation states remain after the last W'
    return nodes, rows


def iteration_transition(nodes, rows):
    """Append local optimizer steps and a prefix of the next iteration.

    Successful post-validation path only. O=1 is an illustrative duration;
    reductions, validation and communication latency are not timed here.
    Repeating the entire dependency-checked schedule preserves its edges.
    """
    optimizer_cost = 1
    period = max(nodes[row[-1]]['end'] - nodes[row[0]]['start']
                 for row in rows) + optimizer_cost
    cutoff = period + max(nodes[row[0]]['start'] for row in rows) + 6
    result = list(nodes)
    for rank, row in enumerate(rows):
        last = nodes[row[-1]]
        next_start = nodes[row[0]]['start'] + period
        assert last['ops'][0][0] == 'W'
        assert last['end'] + optimizer_cost <= next_start
        result.append(dict(rank=rank, ops=[('O', last['ops'][0][1], -1)],
                           start=last['end'], end=last['end']+optimizer_cost,
                           duration=optimizer_cost))
    for node in nodes:
        if node['end'] + period <= cutoff:
            result.append(dict(node, start=node['start']+period,
                               end=node['end']+period, next_iteration=True))
    return result, period, cutoff


def render(name, data, nodes, rows):
    title, subtitle = TITLES[name]
    end = max(n['end'] for n in nodes)
    with_optimizer = name in {'zb', 'zb2p', 'zbv'}
    if with_optimizer:
        nodes, period, end = iteration_transition(nodes, rows)
    paired = name.startswith('dualpipe')
    basic = name in {'naive', 'gpipe', 'onef1b'}
    single_row = name in {'vpp', 'zbv', 'dualpipe', 'dualpipev'}
    # Interleaved tasks share one physical-device row; shade identifies chunks.
    stages = [[s for s, r in enumerate(data['placement']) if r == rank]
              for rank in range(len(rows))]
    x0 = 58 if basic else (78 if single_row else 102)
    unit, rh, gap = min(26, 780 / end), 27, 13
    width = x0 + end * unit + 22
    y0 = 115 if with_optimizer or paired else 96
    ys, rank_bounds, y = {}, [], y0
    for stage_ids in stages:
        start = y
        for s in stage_ids:
            ys[s] = y
            if not single_row:
                y += rh
        if single_row:
            y += rh
        rank_bounds.append((start, y - 3))
        y += gap
    height = y + 54
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)} 多 stage 的 micro-batch 前后向时间线">',
             '<style>text{font-family:system-ui,sans-serif;fill:#334155}</style>',
             f'<rect width="{width}" height="{height}" fill="#fff"/>']
    def text(x, yy, content, size=13, anchor='start', weight='400', fill='#334155'):
        parts.append(f'<text x="{x}" y="{yy}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" style="fill:{fill}">{escape(str(content))}</text>')
    def rect(x, yy, w, h, fill, stroke='none', sw=1, radius=2):
        parts.append(f'<rect x="{x}" y="{yy}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    text(14, 25, title, 17, weight='600')
    text(width - 18, 25, subtitle, 11 if name == 'naive' else 13, anchor='end', fill='#64748b')
    keys = [('F','F 前向'),('B','B 完整反向')] if basic or name=='vpp' else [('F','F 前向'),('X','X 输入梯度'),('W','W 参数梯度')]
    if paired: keys=[('F','F 前向'),('B','B 完整反向'),('X','X 输入梯度'),('W','W 参数梯度')]
    if with_optimizer: keys.append(('O', 'O 优化器更新'))
    vpp_colors = {('F', 0): ('#dfe7ff', '#3f56a4'),
                  ('F', 1): ('#526abd', '#ffffff'),
                  ('B', 0): ('#d9eee3', '#247249'),
                  ('B', 1): ('#318362', '#ffffff'),
                  ('X', 0): ('#c9e9f3', '#236d83'),
                  ('X', 1): ('#287c94', '#ffffff'),
                  ('W', 0): ('#ffebc4', '#956a20'),
                  ('W', 1): ('#9b691f', '#ffffff')}
    if name == 'vpp':
        for i, (kind, chunk) in enumerate(k for k in vpp_colors if k[0] in {'F', 'B'}):
            x = 14 + i * 175
            fill, ink = vpp_colors[(kind, chunk)]
            rect(x,42,18,14,fill,COLORS[kind][1])
            text(x+25,54,f'{kind} {"前向" if kind == "F" else "反向"} · chunk {chunk}',12)
    else:
        for i,(k,label) in enumerate(keys):
            x=14+i*145
            if (name == 'zbv' or paired) and k != 'O':
                rect(x,42,9,14,COLORS[k][0],COLORS[k][1])
                rect(x+9,42,9,14,vpp_colors[(k,1)][0],COLORS[k][1])
            else:
                rect(x,42,18,14,COLORS[k][0],COLORS[k][1])
            text(x+25,54,label,12)
    if with_optimizer:
        text(14,75,('浅色 = chunk 0；深色 = chunk 1；' if name == 'zbv' else '') +
             'O 后编号重新从 1 开始：下一迭代',12,fill='#64748b')
    if paired:
        text(14,75,('浅色 = 副本 A；深色 = 副本 B；' if name == 'dualpipe' else
                    '浅色 = chunk 0；深色 = chunk 1；') +
             '黑框 = F+B 配对；斜纹 = 重叠区间',12,fill='#64748b')
    text(14,y0-21,'数字 = micro-batch 编号；空白 = 等待',10 if name == 'naive' else 12,fill='#64748b')
    text(width-18,y0-21,'时间 →',12,anchor='end',fill='#64748b')
    for rank, ((top,bottom), stage_ids) in enumerate(zip(rank_bounds,stages)):
        rect(6,top-4,width-14,bottom-top+9,'#f8fafc',radius=4)
        text(13,(top+bottom)/2+4,f'S{rank}' if basic else f'GPU {rank}',12,weight='600')
        for stage in stage_ids:
            if basic or single_row:
                continue
            label=f'S{stage}'
            if paired and name=='dualpipe': label=('A' if stage<4 else 'B')+f'·S{stage%4}'
            text(x0-9,ys[stage]+17,label,12,anchor='end')
    ticks = [t for t in range(0, math.ceil(end), 4) if end - t >= 2]
    for t in ticks + [end]:
        x=x0+t*unit
        parts.append(f'<path d="M{x} {y0-8} V{y-gap+2}" stroke="#e2e8f0" stroke-width="1"/>')
        text(x,y+15,f'{t:g}',11,anchor='middle',fill='#64748b')
    for n in nodes:
        x=x0+n['start']*unit+1
        w=n['duration']*unit-2
        if paired and len(n['ops']) == 2:
            # A single physical-rank lane: the F-only, overlapping and B-only
            # intervals share its full height. Hatching encodes concurrent work.
            forward = next(op for op in n['ops'] if op[0] == 'F')
            backward = next(op for op in n['ops'] if op[0] == 'B')
            _, fs, fm = forward
            _, bs, bm = backward
            ff, fi = vpp_colors[('F', stages[n['rank']].index(fs))]
            bf, bi = vpp_colors[('B', stages[n['rank']].index(bs))]
            yy = ys[fs]
            f_end = x0 + (n['start'] + data['costs']['F']) * unit
            b_start = x0 + (n['end'] - data['costs']['B']) * unit
            assert x < b_start < f_end < x+w
            pattern = f'overlap-{n["rank"]}-{fm}-{bm}'
            parts.append(f'<defs><pattern id="{pattern}" width="6" height="6" '
                         f'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
                         f'<rect width="6" height="6" fill="{ff}"/>'
                         f'<rect width="3" height="6" fill="{bf}"/></pattern></defs>')
            rect(x,yy,b_start-x,rh-5,ff,radius=0)
            rect(b_start,yy,f_end-b_start,rh-5,f'url(#{pattern})',radius=0)
            rect(f_end,yy,x+w-f_end,rh-5,bf,radius=0)
            rect(x,yy,w,rh-5,'none','#1e293b',sw=1.3,radius=2)
            text((x+b_start)/2,yy+15,str(fm+1),11,anchor='middle',fill=fi)
            text((f_end+x+w)/2,yy+16,str(bm+1),14,anchor='middle',fill=bi)
            continue
        for kind,stage,micro in n['ops']:
            yy=ys[stage]
            # Show the nominal F/B durations inside a reserved overlap group.
            # F occupies its first unit; B overlaps F for 0.5 units and runs
            # for two units. Downstream group scheduling waits for the frame.
            op_x, op_w = x, w
            if len(n['ops']) == 2:
                offset = 0 if kind == 'F' else n['duration'] - data['costs'][kind]
                op_x = x + offset * unit
                op_w = data['costs'][kind] * unit - 2
            fill,stroke=COLORS[kind]
            ink = stroke
            if single_row and kind != 'O':
                chunk = stages[n['rank']].index(stage)
                fill, ink = vpp_colors[(kind, chunk)]
            rect(op_x,yy,op_w,rh-5,fill,stroke,sw=.7)
            label = 'O' if kind == 'O' else (f'{kind}{micro+1}' if basic else str(micro+1))
            text(op_x+op_w/2,yy+16,label,12 if basic else 14,anchor='middle',fill=ink)
        if len(n['ops'])==2:
            top,bottom=rank_bounds[n['rank']]
            rect(x-1,top-2,w+2,bottom-top+4,'none','#1e293b',sw=1.6,radius=3)
    note = 'F = 1，完整 B = 2；省略通信与参数更新。'
    if with_optimizer: note = 'F = X = W = 1；O = 1（示意）；展示后置校验通过的路径，未展开通信。'
    if paired: note = 'F = 1，B = 2，X = W = 1；配对重叠 0.5，共占 2.5（示意）。'
    text(14,height-12,note,10 if name == 'naive' else 12,fill='#64748b')
    parts.append('</svg>')
    filenames = {'naive': '04-pp-figure-02.svg', 'gpipe': '04-pp-figure-03.svg',
                 'onef1b': '04-pp-figure-04.svg'}
    target=ASSETS/filenames.get(name, f'04-pp-timeline-{name}.svg')
    target.write_text('\n'.join(parts)+'\n')
    return target


def main():
    all_data={name: basic_schedule(name) for name in ['naive', 'gpipe', 'onef1b']}
    all_data.update(json.loads(DATA.read_text())['schedules'])
    for name, data in all_data.items():
        nodes, rows=schedule(data)
        output=render(name,data,nodes,rows)
        busy=[sum(nodes[i]['duration'] for i in row) for row in rows]
        print(f'{name}: {sum(len(n["ops"]) for n in nodes)} operations; '
              f'end={max(n["end"] for n in nodes)}; busy={busy}; dependencies OK; {output.name}')

if __name__=='__main__':
    main()
