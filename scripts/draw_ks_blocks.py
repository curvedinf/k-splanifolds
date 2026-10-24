"""Build the KS1/KS2/KS3 notebook with the RHAC PyNCD diagram toolchain.

Python >=3.13, a PyNCD checkout on PYTHONPATH, and a built tsncd checkout
selected by TSNCD_DIST are required for regeneration. See the notebook's
last section for pinned setup commands. No Torch import or GPU is needed.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import textwrap

import construction_helpers  # noqa: F401; enables PyNCD composition operators
import agent_display
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as term
import notebooks.display.notebook_diagrams as diagrams
from notebooks.classic.shared_mechanisms import contract
from notebooks.sota.DeepSeekV41Flash.construction_idioms import hold, over, route
from websocket_transfer.headless import HeadlessRenderer, find_dist


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = '198183cbd21e212e6ad9cc11030979836bda6020'
RHAC_REVISION = 'd093ec868246ba86155a7f9d8039c4bec61d9e51'
PYNCD_REVISION = 'a051f8f2df12be97bb438a7b19980513fbe11c36'
TSNCD_REVISION = '70034600050b8297a3fe91ca7b6cc75c1d26bbc2'
SOURCE_BASE = f'https://github.com/curvedinf/k-splanifolds/blob/{SOURCE_REVISION}/'
REFERENCE = f'https://github.com/curvedinf/SDGraft/tree/{RHAC_REVISION}/RHAC/docs/diagrams'
SOURCES = (
    'experiments/common/splanifold.py',
    'experiments/denoising/run.py',
    'experiments/common/ks2.py',
    'experiments/common/ks2_stabilized.py',
    'experiments/common/ks3.py',
)

REAL = cat.Reals()
D = cat.RawAxis.named('d', code_form='input_features')
E = cat.RawAxis.named('e', code_form='output_features')
I = cat.RawAxis.named('i', code_form='input_coordinates_K')
O = cat.RawAxis.named('o', code_form='output_coordinates_N')
S = cat.RawAxis.named('s', code_form='subspaces_S')
R = cat.RawAxis.named('r', code_form='coordinates_per_subspace_R')
T = term.DynamicName('t', code_form='mean_coordinate_singleton').capture(
    cat.RawAxis(_size=nm.Integer(1)))
B = term.DynamicName('b', code_form='spine_basis_terms').capture(
    cat.RawAxis(_size=nm.Integer(6)))


def array(*axes):
    return cat.Array(REAL, axes)


BLUE, GREEN, PURPLE, ORANGE = '#DCEBFA', '#DDF0E7', '#E9E1F5', '#FFE9CD'


def panel(body, title, description, formula='', color=BLUE, source=None):
    refs = () if source is None else (
        cat.CodeReference(label=source, url=SOURCE_BASE + source),)
    return cat.Block.template(
        body, title=r'\text{' + title + '}', description=description,
        formula=formula, fill_color=color, references=refs)


def operation(name, inputs, outputs, title, description, formula='', color=BLUE):
    """An explicitly typed, documented operation; not a numerical kernel."""
    op = cat.Broadcasted(
        operator=ops.GenericOperator(name=term.DynamicName(name)),
        input_weaves=tuple(cat.Weave(REAL, shape) for shape in inputs),
        output_weaves=tuple(cat.Weave(REAL, shape) for shape in outputs),
        reindexings=tuple(cat.ProdObject().identity() for _ in inputs))
    return panel(op, title, description, formula, color)


def ks1_map():
    coordinates = operation(
        r'\mathrm{Coords}_1', ((I,),), ((T,), (I,), (I,)),
        'KS1 coordinates',
        'Apply extrapolation, then compute the mean, deviations and signed '
        'coordinate weights. Only an exactly zero coordinate sum uses uniform '
        'weights. Outputs are t, delta and w, in that order.',
        r'u=(1+2\epsilon)r-\epsilon,\quad t=\tfrac1K\sum_i u_i,\quad '
        r'\delta_i=u_i-t,\quad w_i=u_i/\sum_j u_j', GREEN)
    spine = operation(
        r'\mathrm{Spine}_1', ((T,), (I,)), ((O,),),
        'Hermite spine',
        'The anchors are a0 and a1. The position-tangent control points P0 and P1 '
        'produce endpoint tangents v0 and v1 using the coordinate weights. '
        'H is the cubic Hermite interpolant, with arguments (p0,p1,v0,v1;t).',
        r'v_{q,o}=\sigma\sum_i w_i(P_{q,i,o}-a_{q,o}),\quad '
        r'\mathrm{spine}=H(a_0,a_1,v_0,v_1;t)', BLUE)
    displacement = operation(
        r'\mathrm{Disp}_1', ((T,), (I,)), ((O,),),
        'Hermite displacement',
        'Deviations contract with endpoint basis vectors B0/B1 and their '
        'tangent control points Q0/Q1. This is a second Hermite interpolant, '
        'not the damped factorized edge construction used by KS2.',
        r'd_{q,o}=\sum_i\delta_i B_{q,i,o},\quad '
        r'e_{q,o}=\sigma\sum_i\delta_i(Q_{q,i,o}-B_{q,i,o}),\quad '
        r'\mathrm{disp}=H(d_0,d_1,e_0,e_1;t)', PURPLE)
    body = (coordinates
            @ route((0, 2, 0, 1), (array(T), array(I), array(I)))
            @ (spine * displacement) @ ops.AdditionOp.template())
    return panel(body, 'KS1: original splanifold map',
                 'The repository calls this function splanifold_map. Input r has K '
                 'coordinates and output has N coordinates. The map itself has '
                 'no learned projection or residual bypass. Batch axes are omitted.',
                 r'f_1(r)=\mathrm{spine}(r)+\mathrm{disp}(r)',
                 source=SOURCES[0])


def ks1_block():
    squash = operation(
        r'\mathrm{Unit}', ((R,),), ((R,),), 'Bound map inputs',
        'The denoiser maps each projected coordinate into (0,1) before KS1.',
        r'r=(1+\tanh(c))/2', GREEN)
    geometry = operation(
        r'f_{1,s}', ((R,),), ((R,),), 'KS1 map in each chart',
        'Each chart owns separate anchors, bases and tangent control points. '
        'The map above is evaluated with K=N=8, sigma=1 and extrapolation=0.',
        r'\Delta_s=f_{1,s}((1+\tanh(c_s))/2)', PURPLE)
    chart = (route((0, 0), (array(R),))
             @ (hold(array(R)) * (squash @ geometry))
             @ ops.AdditionOp.template())
    body = (ops.Linear.template((D,), (S, R), r'W_{\rm down}')
            @ over((S,), chart)
            @ ops.Linear.template((S, R), (E,), r'W_{\rm up}'))
    return panel(body, 'KS1: denoising wrapper',
                 'Exact wrapper structure of KS1Denoiser: d=e=169, rank R=8, '
                 'configurable chart count S. Both projections have no bias. '
                 'The bypass carries raw projected coordinates c, not the '
                 'squashed map inputs. It is not an outer x+y residual.',
                 r'c=\operatorname{reshape}(W_{\rm down}x),\quad '
                 r'y=W_{\rm up}\operatorname{vec}(c+f_1((1+\tanh c)/2))',
                 source=SOURCES[1])


def ks2_map():
    basis = operation(
        r'\Phi', ((I,),), ((B,),), 'Six spine basis terms',
        'Take t=mean(x). The first four terms are the unbounded cubic Hermite '
        'basis h00,h01,h10,h11. The last two are T2(z) and T3(z), with '
        'z=tanh((softplus(a)+1e-4)*(t+0.5*tanh(b))).',
        r'\Phi(t)=[h_{00},h_{01},h_{10},h_{11},T_2(z),T_3(z)]', GREEN)
    edge_x = operation(
        r'E_x', ((I,),), ((I, O),), 'Coordinate edge factor',
        'Warp each input coordinate with a second learned scale and shift. '
        'Evaluate T0,T1,T2,T3 and contract their four coefficients separately '
        'for each input/output coordinate pair. These weights are not shared '
        'between input coordinates.',
        r'X_{io}=\sum_{a=0}^{3}T_a(z_i)C^x_{aio}', PURPLE)
    damping = operation(
        r'\mathrm{Den}_2', ((I,),), ((I, O),), 'Original KS2 damping',
        'Let total=sum(x), t=mean(x), delta=x-t. Use w=x/total unless '
        'abs(total)<1e-6, when w=1/K. Both damping strengths use softplus. '
        'This is the original ks2.py denominator, not the stabilized variant.',
        r'D_{io}=1+\operatorname{softplus}(a_{io})|\delta_i|+'
        r'\operatorname{softplus}(b_{io})w_i^2', ORANGE)
    divide = operation(
        r'\mathrm{Divide}', ((I, O), (I, O)), ((I, O),),
        'Damp each edge', 'Elementwise division before reducing input coordinates.',
        r'E_{io}=L_{io}X_{io}/D_{io}', ORANGE)
    factors = (basis @ route((0, 0), (array(B),))
               @ (ops.Linear.template((B,), (O,), r'C_{\rm spine}')
                  * ops.Linear.template((B,), (I, O), r'C_t')))
    edges = (contract(((I, O), (I, O)), (I, O)) * hold(array(I, O)))
    edges = edges @ divide @ contract(((I, O),), (O,))
    body = (route((0, 0, 0), (array(I),))
            @ (factors * edge_x * damping)
            @ (hold(array(O)) * edges) @ ops.AdditionOp.template())
    return panel(body, 'KS2: spine and damped factorized edges',
                 'Standalone KS2 maps K input coordinates directly to N outputs. '
                 'C_spine has shape [6,N], C_t [6,K,N], and C_x [4,K,N]. '
                 'The spine and longitudinal edge factor share Phi and its warp. '
                 'There is no projection, coordinate bypass, ReLU or iteration.',
                 r'f_{2,o}(x)=\sum_b\Phi_b C^{\rm spine}_{bo}+'
                 r'\sum_i\frac{(\sum_b\Phi_b C^t_{bio})X_{io}}{D_{io}}',
                 source=SOURCES[2])


def ks2_block(geo_steps=2):
    if not isinstance(geo_steps, int) or isinstance(geo_steps, bool) or geo_steps < 1:
        raise ValueError('geo_steps must be a positive integer')
    geometry = operation(
        r'G_{2,s}', ((R,),), ((R,),), 'One KS2 geometry evaluation',
        'Same equations as the direct map, with K=N=R and independent parameters '
        'per chart. The same chart parameters are reused at every geometry step.',
        r'c_s^{(j+1)}=c_s^{(j)}+G_{2,s}(c_s^{(j)})', PURPLE)
    step = (route((0, 0), (array(R),)) @ (hold(array(R)) * geometry)
            @ ops.AdditionOp.template())
    refinement = step
    for _ in range(geo_steps - 1):
        refinement = refinement @ step
    body = (ops.Linear.template((D,), (S, R), r'W_{\rm down}')
            @ over((S,), refinement)
            @ ops.Linear.template((S, R), (E,), r'W_{\rm up}'))
    return panel(body, f'KS2Cell: {geo_steps} shared-parameter geometry steps',
                 'The default KS2Cell has S=4, R=8 and geo_steps=2. Each '
                 'addition is inside chart space. Geometry parameters differ '
                 'between charts but are shared between steps. Projections '
                 'have no bias. No outer residual or ReLU is included.',
                 r'c^{(0)}=\operatorname{reshape}(W_{\rm down}x),\quad '
                 r'c^{(j+1)}=c^{(j)}+G_2(c^{(j)}),\quad '
                 r'y=W_{\rm up}\operatorname{vec}(c^{(L)})',
                 source=SOURCES[2])


def ks3_block():
    clip = operation(
        r'\mathrm{Clamp}', ((S, R),), ((S, R),), 'Promote and clamp coordinates',
        'After the down projection, geometry is FP32 for ordinary/FP16/BF16 '
        'inputs, or FP64 for a double model. Clamp coordinates to [-65504,65504]. '
        'The bypass below carries these clamped coordinates.',
        r'c=\operatorname{clip}(W_{\rm down}x,-65504,65504)', GREEN)
    mean = operation(
        r'\mathrm{MeanCap}', ((R,),), ((T,),), 'Bounded chart mean',
        'Reduce only the coordinate axis within each chart, then apply the '
        'smooth cap. For rank one this is 4*tanh(c/4).',
        r't_s=4\tanh\left(\frac{1}{4R}\sum_r c_{sr}\right)', GREEN)
    cubic = operation(
        r'H_{\theta_s}', ((T,),), ((R,),), 'Hermite cubic in Horner form',
        'Four trainable Hermite coefficients per output coordinate, independently '
        'for each chart. Horner evaluation preserves the Hermite parameterization. '
        't is bounded to (-4,4), not to the interpolation interval [0,1].',
        r'H_{sr}=(( (2p_0-2p_1+p_2+p_3)t+'
        r'(-3p_0+3p_1-2p_2-p_3))t+p_2)t+p_0', BLUE)
    warp = operation(
        r'\mathrm{Warp}_s', ((T,),), ((T,),), 'Learned chart warp',
        'One raw scale and shift per chart; the resulting warped scalar is '
        'shared by all output coordinates in that chart.',
        r'z_s=\tanh[(\operatorname{softplus}(a_s)+10^{-4})'
        r'(t_s+\tfrac12\tanh b_s)]', GREEN)
    chebyshev = operation(
        r'C_{\theta_s}', ((T,),), ((R,),), 'Two warped Chebyshev terms',
        'The remaining two coefficients p4 and p5 are independent per chart and '
        'output coordinate. T2(z)=2z^2-1 and T3(z)=4z^3-3z.',
        r'C_{sr}=p_{4,sr}T_2(z_s)+p_{5,sr}T_3(z_s)', PURPLE)
    activate = operation(
        r'\mathrm{ReLU}^2', ((R,),), ((R,),), 'Squared rectifier',
        'Apply ReLU and square after adding the clamped coordinate bypass, '
        'the Hermite cubic and the two warped terms.',
        r'h_{sr}=\max(0,c_{sr}+H_{sr}+C_{sr})^2', ORANGE)
    spine = (mean @ route((0, 0), (array(T),))
             @ (cubic * (warp @ chebyshev)) @ ops.AdditionOp.template())
    chart = (route((0, 0), (array(R),)) @ (hold(array(R)) * spine)
             @ ops.AdditionOp.template() @ activate)
    body = (ops.Linear.template((D,), (S, R), r'W_{\rm down}')
            @ clip @ over((S,), chart)
            @ ops.Linear.template((S, R), (E,), r'W_{\rm up}'))
    return panel(body, 'KS3: bounded spine cell',
                 'KS3 supports any positive chart rank R, including the selected '
                 'rank-one configuration. It has no KS2 edges or edge damping '
                 'and no repeated geometry loop. Features are flattened and '
                 'cast to up.weight.dtype before the up projection. Normalization, '
                 'attention, time mixing and the outer residual are outside KS3.',
                 r'y=W_{\rm up}\operatorname{vec}('
                 r'\operatorname{ReLU}(c+H_\theta(t)+C_\theta(z))^2)',
                 source=SOURCES[4])


CAPTIONS = {
    'ks1-map': 'KS1 core: a Hermite spine plus a Hermite displacement.',
    'ks1-block': 'KS1Denoiser: projected rank-eight charts, bounded map input, coordinate bypass.',
    'ks2-map': 'Direct KS2: shared spine basis, two edge factors, damping, then sum over inputs.',
    'ks2-block': 'KS2Cell: down projection, two refinements with shared parameters, up projection.',
    'ks3-block': 'KS3: bounded chart mean, cubic and warped terms, coordinate bypass, ReLU squared.',
}


def build_figures():
    return dict(zip(CAPTIONS, (ks1_map(), ks1_block(), ks2_map(), ks2_block(), ks3_block())))


def validate_signatures(figures):
    for name, figure in figures.items():
        expected = ([array(I)], [array(O)]) if name.endswith('-map') else ([array(D)], [array(E)])
        actual = (list(figure.dom()), list(figure.cod()))
        if actual != expected:
            raise ValueError(f'{name}: expected {expected}, got {actual}')


def write_notebook(output):
    import nbformat as nbf

    cells = []

    def md(source):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))

    def code(source):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))

    def figure(name):
        code(f'await show_figure("{name}")')

    md(r'''
        # KS1, KS2 and KS3 — neural circuit diagrams

        Five source-matched diagrams using the **same PyNCD / tsncd system as
        [RHAC](REFERENCE_LINK)**. Saved inline figures are included; opening this
        notebook does not require PyNCD, Torch, a GPU, or a network connection.
        Running the default cells requires IPython and the PNGs in this folder.
        Set `RENDER = True` after the setup at the end to redraw from the editable
        definitions in [`draw_ks_blocks.py`](../../scripts/draw_ks_blocks.py).

        **Reading the diagrams.** One wire is one tensor axis. Dashed separators
        divide arrays; branching copies an array; contraction joins and sums an
        axis. A shared axis retained in a contraction denotes a pointwise product.
        Batch/token axes are omitted. Colored boxes are documented operations,
        not executable numerical kernels. Learned coefficients belong to the boxes;
        separate wires for parameters are omitted. Chart-indexed boxes have separate
        parameters for each chart. The diagrams do not claim training
        quality, speed, or quantization equivalence.

        **Axes:** `d/e` = input/output feature widths; `i/o` = map input/output
        coordinates (`K/N`); `s/r` = chart count/rank (`S/R`); `t` = the singleton
        mean coordinate; `b` = six spine terms. The projected wrappers show reshaping
        into `[S,R]` as part of their dense projections.

        For hover explanations, formulas and source links, open
        [the interactive companion](comparison/index.html). It also offers
        box-and-arrow views of the same graphs.
    '''.replace('REFERENCE_LINK', REFERENCE))
    code('''
        from pathlib import Path
        from IPython.display import Image, display

        RENDER = False  # True: redraw with PyNCD; False: display checked-in PNGs.
        root = next((p for p in (Path.cwd(), *Path.cwd().parents)
                     if (p / "scripts/draw_ks_blocks.py").is_file()), None)
        if root is None:
            raise FileNotFoundError("Run from this repository or docs/diagrams.")
        assets = root / "docs/diagrams"

        if RENDER:
            import sys
            sys.path.insert(0, str(root / "scripts"))
            from draw_ks_blocks import build_figures, validate_signatures
            import notebooks.display.notebook_diagrams as diagrams
            from websocket_transfer.headless import HeadlessRenderer
            figures = build_figures()
            validate_signatures(figures)
            settings = diagrams.DiagramSettings(
                dark_mode=diagrams.ColorMode.LIGHT,
                advanced_display=diagrams.AdvancedDisplay.LEGEND,
                sub_blocks=diagrams.SubBlocks.NO_BODIES, width=950)

        async def show_figure(name):
            if RENDER:
                message = diagrams.sent_figure(figures[name], settings).message()
                async with HeadlessRenderer(scale=2) as renderer:
                    await renderer.render_json(message["data"], message["settings"],
                                               message.get("auxiliary"))
                    png = await renderer.capture_rendered(format="png")
            else:
                png = (assets / f"{name}.png").read_bytes()
            display(Image(data=png, width=1100))
    ''')
    md(r'''
        ## KS1 — the original splanifold map

        Source: [`splanifold_map`](../../experiments/common/splanifold.py).
        There is no `KS1` class in the common module: this function is the original
        geometric map used by the KS1 experiments. It maps `(..., K)` to `(..., N)`.

        With extrapolation $epsilon$ and tangent scale $sigma$:
        $$u=(1+2\epsilon)r-\epsilon,\quad t=\frac1K\sum_i u_i,\quad
        \delta_i=u_i-t,\quad
        w_i=\begin{cases}u_i/\sum_j u_j&\sum_j u_j\ne0\\1/K&\text{otherwise.}\end{cases}$$
        Write $H(p_0,p_1,v_0,v_1;t)=h_{00}p_0+h_{01}p_1+h_{10}v_0+h_{11}v_1$,
        where $h_{00}=2t^3-3t^2+1$, $h_{01}=-2t^3+3t^2$,
        $h_{10}=t^3-2t^2+t$ and $h_{11}=t^3-t^2$.
        For endpoint $q\in\{0,1\}$, the controls give
        $$v_q=\sigma\sum_i w_i(P_{q,i}-a_q),\quad
        d_q=\sum_i\delta_i B_{q,i},\quad
        e_q=\sigma\sum_i\delta_i(Q_{q,i}-B_{q,i}).$$
        The map is $f_1(r)=H(a_0,a_1,v_0,v_1;t)+H(d_0,d_1,e_0,e_1;t)$.
        `P` denotes position-tangent control points; `B` endpoint bases; `Q`
        basis-tangent control points. The function takes these controls as arguments.
    ''')
    figure('ks1-map')
    md(r'''
        ### KS1 as a projected block

        Source: [`KS1Denoiser`](../../experiments/denoising/run.py).
        This concrete wrapper fixes input/output width to **169** and rank to **8**;
        its chart count is configurable. It learns the chart controls and two
        bias-free projections. It bounds the input to the geometric map, then adds
        the map's result to the **unsquashed** projected coordinates:
        $$c=\operatorname{reshape}(W_{\rm down}x),\quad
        y=W_{\rm up}\operatorname{vec}\left[c+f_1\left((1+\tanh c)/2\right)\right].$$
        Here $\sigma=1$, $\epsilon=0$. This wrapper is separate from the core map;
        it has one chart-space addition and no outer input residual or ReLU.
    ''')
    figure('ks1-block')
    md(r'''
        ## KS2 — learned spine and factorized edges

        Source: [`KS2`](../../experiments/common/ks2.py). The direct map takes
        `(..., K)` to `(..., N)` in one geometric evaluation. Let
        $$t=\operatorname{mean}(x),\quad
        z(v;a,b)=\tanh[(\operatorname{softplus}(a)+10^{-4})(v+\tfrac12\tanh b)].$$
        Its six-term spine basis is
        $\Phi(t)=[h_{00}(t),h_{01}(t),h_{10}(t),h_{11}(t),T_2(z(t)),T_3(z(t))]$.
        A separate warp gives the coordinate basis $[1,z_i,T_2(z_i),T_3(z_i)]$.
        $$L_{io}=\sum_b\Phi_b C^t_{bio},\quad
        X_{io}=\sum_{a=0}^3T_a(z_i)C^x_{aio},\quad
        f_{2,o}(x)=\sum_b\Phi_b C^{\rm spine}_{bo}+\sum_i\frac{L_{io}X_{io}}{D_{io}}.$$
        The original denominator is
        $$D_{io}=1+\operatorname{softplus}(\alpha_{io})|x_i-t|
        +\operatorname{softplus}(\beta_{io})w_i^2,$$
        with $w=x/\sum_i x_i$ except that $|\sum_i x_i|<10^{-6}$ gives $w_i=1/K$.
        These are signed weights, not a softmax. The Hermite coordinate is unbounded
        in this original implementation; only the Chebyshev coordinate is warped.
    ''')
    figure('ks2-map')
    md(r'''
        ### KS2Cell — projections and repeated geometry

        Source: [`KS2Cell`](../../experiments/common/ks2.py). The direct map above
        has no projections or loop. The cell adds them:
        $$c^{(0)}=\operatorname{reshape}(W_{\rm down}x),\quad
        c^{(j+1)}=c^{(j)}+G_2(c^{(j)}),\quad
        y=W_{\rm up}\operatorname{vec}(c^{(L)}).$$
        The diagram uses the default **two steps**, with the **same parameters** in
        both steps. Charts own separate parameters. Defaults are $S=4$, $R=8$;
        all these sizes and `geo_steps` are configurable. There is no ReLU or outer
        input residual in this cell.
    ''')
    figure('ks2-block')
    md(r'''
        **Stabilized KS2 is a separate implementation.**
        [`ks2_stabilized.py`](../../experiments/common/ks2_stabilized.py) preserves
        the spine/edge structure and parameter layout, but clamps coordinates,
        computes geometry in FP32, uses $t=4\tanh(\operatorname{mean}(x)/4)$,
        and replaces the denominator predicates with normalized squared deviation
        and coordinate energy. Its damping strengths use a bounded scaled sigmoid;
        `KS2StabilizedCell` defaults to one step. The original KS2 diagrams above
        should not be read as the stabilized numerical formulas.

        ## KS3 — bounded spine, coordinate bypass, ReLU²

        Source: [`KS3`](../../experiments/common/ks3.py). After the bias-free down
        projection and reshape, coordinates are promoted for geometry and clamped
        to $[-65504,65504]$. In each chart:
        $$t_s=4\tanh(\operatorname{mean}_r(c_{sr})/4),\quad
        z_s=z(t_s;a_s,b_s),$$
        $$H_{sr}=h_{00}(t_s)p_{0,sr}+h_{01}(t_s)p_{1,sr}
        +h_{10}(t_s)p_{2,sr}+h_{11}(t_s)p_{3,sr},$$
        $$C_{sr}=p_{4,sr}T_2(z_s)+p_{5,sr}T_3(z_s),\quad
        y=W_{\rm up}\operatorname{vec}\left[\operatorname{ReLU}(c+H+C)^2\right].$$
        The code evaluates both polynomials in Horner form. There are six coefficients
        per chart/output coordinate, plus one raw scale and shift per chart. For
        rank $R>1$, output coordinates share the chart mean and warp, but have
        separate coefficients and retain their own coordinate bypasses.

        KS3 has **no edge terms, edge damping, or geometry loop**. Its bounded mean
        lies in $(-4,4)$, not $[0,1]$. Normalization, attention, time mixing and the
        outer residual belong to the enclosing model. The figure describes the
        floating-point reference; [`KS3Int8`](../../experiments/common/ks3_int8.py)
        adds quantized projections and a distinct precision/training path.
    ''')
    figure('ks3-block')
    md(r'''
        ## Compare the boundaries

        Here $D_{\rm in/out}$ are feature widths; $K,N$ are direct-map dimensions;
        $S,R$ are chart count and rank. Control counts assume all displayed geometric
        controls are learned. Projection biases are absent.

        | Construction | Geometric controls / total parameters | Internal bypass | Activation |
        |---|---|---|---|
        | KS1 direct map | $2N+6KN$ controls; $\sigma,\epsilon$ supplied separately | None | None |
        | KS1Denoiser | $S[R(D_{\rm in}+D_{\rm out})+2R+6R^2]$, with widths 169 and $R=8$ | Once, in chart space | Tanh on map input only |
        | KS2 direct map | $6N+12KN+4$ | None | Tanh warps only |
        | KS2Cell | $S[R(D_{\rm in}+D_{\rm out})+6R+12R^2+4]$ | Every shared geometry step | Tanh warps only |
        | KS3 | $S[R(D_{\rm in}+D_{\rm out}+6)+2]$ | Once, before ReLU² | Bounded mean, tanh warp, ReLU² |

        Increasing KS2 `geo_steps` reuses weights and does not increase parameter
        count. The README's $D_{\rm in}=D_{\rm out}=96$, $S=368$, $R=1$ KS3 example
        has **73,600** parameters. Equal feature widths do not imply equal capacity.
    ''')
    code('''
        # Arithmetic check for the documented KS3 configuration; no model allocation.
        d_in, d_out, charts, rank = 96, 96, 368, 1
        parameters = charts * (rank * (d_in + d_out + 6) + 2)
        assert parameters == 73_600
        print(f"KS3 (96 -> 96, 368 charts, rank 1): {parameters:,} parameters")
    ''')
    md(r'''
        ## Regenerate or edit the diagrams

        The default display path only needs `ipython` and the local PNGs. To edit
        the diagrams, use Python **3.13+** and Node/npm in an isolated tooling
        environment. The pinned revisions below match the RHAC reference:

        ```bash
        mkdir -p /tmp/ks-diagram-tools
        cd /tmp/ks-diagram-tools
        git clone https://github.com/mit-zardini-lab/pyncd.git
        git -C pyncd checkout PYNCD_PIN
        git clone https://github.com/mit-zardini-lab/tsncd.git
        git -C tsncd checkout TSNCD_PIN
        (cd tsncd && npm ci && npm run build)
        python3.13 -m venv venv
        venv/bin/python -m pip install websockets playwright ipython pandas pillow nbformat nbclient ipykernel
        venv/bin/python -m playwright install chromium
        export PYTHONPATH=/tmp/ks-diagram-tools/pyncd
        export TSNCD_DIST=/tmp/ks-diagram-tools/tsncd/dist

        # Substitute the path to this checkout. --check validates axis signatures only.
        venv/bin/python /path/to/k-splanifolds/scripts/draw_ks_blocks.py --check
        venv/bin/python /path/to/k-splanifolds/scripts/draw_ks_blocks.py
        ```

        The generator validates all five input/output signatures, renders PNGs,
        writes readable operation listings and JSON diagrams, builds the interactive
        companion, and executes this notebook with saved outputs. Alternatively,
        start a notebook kernel in this environment with the two environment
        variables above and set `RENDER = True`. Edit the named diagram factories
        in `scripts/draw_ks_blocks.py`, then restart the kernel before rebuilding.

        [`provenance.json`](provenance.json) records tool revisions, source hashes
        and the inspected project revision. Graph signatures and notebook execution
        validate the documentation artifacts; they are not model quality tests.
    '''.replace('PYNCD_PIN', PYNCD_REVISION).replace('TSNCD_PIN', TSNCD_REVISION))
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    })
    for index, cell in enumerate(notebook.cells):
        cell.id = f'ks-blocks-{index:02d}'
    nbf.write(notebook, output / 'ks-blocks.ipynb')


def execute_notebook(output):
    """Execute in the generator's Python, without installing a user kernel."""
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client.kernelspec import KernelSpecManager
    import sys
    import tempfile

    with tempfile.TemporaryDirectory(prefix='ks-notebook-kernel-') as temp:
        kernel = Path(temp) / 'python3'
        kernel.mkdir()
        (kernel / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'KS diagrams validation', 'language': 'python',
        }))
        path = output / 'ks-blocks.ipynb'
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, timeout=120, kernel_name='python3',
                                resources={'metadata': {'path': str(ROOT)}})
        client.create_kernel_manager()
        client.km.kernel_spec_manager = KernelSpecManager(kernel_dirs=[temp])
        client.execute()
        nbformat.validate(notebook)
        nbformat.write(notebook, path)


def revision(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                                   text=True).strip()


async def render(output):
    figures = build_figures()
    validate_signatures(figures)
    output.mkdir(parents=True, exist_ok=True)
    settings = diagrams.DiagramSettings(
        mode=diagrams.DiagramMode.HTML, dark_mode=diagrams.ColorMode.LIGHT,
        advanced_display=diagrams.AdvancedDisplay.INTERACTIVE,
        sub_blocks=diagrams.SubBlocks.NO_BODIES, width=950,
        title='KS1, KS2 and KS3', page_directory=output)
    group = diagrams.PageVariantGroup('block', 'Construction')
    variants = [diagrams.PageVariant(name, group, name.upper(), caption, term=figures[name])
                for name, caption in CAPTIONS.items()]
    await diagrams.show_page_variants(variants, 'KS1, KS2 and KS3', settings=settings,
                                     slug='comparison', initial='ks3-block')
    for html in (output / 'comparison.html', output / 'comparison/index.html'):
        html.write_text('\n'.join(line.rstrip() for line in html.read_text().splitlines()) + '\n')
    static_settings = replace(settings, advanced_display=diagrams.AdvancedDisplay.LEGEND)
    async with HeadlessRenderer(scale=2) as renderer:
        for name, figure in figures.items():
            (output / f'{name}.listing.txt').write_text(agent_display.listing(figure) + '\n')
            message = diagrams.sent_figure(figure, static_settings).message()
            (output / f'{name}.json').write_text(json.dumps(message) + '\n')
            await renderer.render_json(message['data'], message['settings'],
                                       message.get('auxiliary'))
            (output / f'{name}.png').write_bytes(await renderer.capture_rendered(format='png'))
            print(f'Rendered {name}', flush=True)
    write_notebook(output)
    provenance = {
        'reference': REFERENCE, 'inspected_project_revision': SOURCE_REVISION,
        'checkout_revision_at_generation': revision(ROOT),
        'pyncd_commit': revision(Path(construction_helpers.__file__).resolve().parents[1]),
        'tsncd_commit': revision(find_dist().parent),
        'source_sha256': {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                          for path in SOURCES},
        'python_packages': {package: importlib.metadata.version(package)
                            for package in ('websockets', 'playwright', 'ipython', 'pandas',
                                            'pillow', 'nbformat', 'nbclient', 'ipykernel')},
        'figures': list(figures), 'gpu_used': False,
    }
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate graph signatures only.')
    args = parser.parse_args()
    if args.check:
        figures = build_figures()
        validate_signatures(figures)
        for name, figure in figures.items():
            print(name, agent_display.summary(figure))
        return
    output = ROOT / 'docs/diagrams'
    asyncio.run(render(output))
    execute_notebook(output)
    print(f'Executed notebook: {output / "ks-blocks.ipynb"}')


if __name__ == '__main__':
    main()
