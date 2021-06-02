#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cProfile
import datetime
import pprint
import re
import sys
import time

import numpy as np


"""
文字盤マトリックス

' X ' のみ特殊扱い(ブロックがおけない場所)
"""
str_matrix = \
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', ' X ',
     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', ' X ',
     '  1', '  2', '  3', '  4', '  5', '  6', '  7',
     '  8', '  9', ' 10', ' 11', ' 12', ' 13', ' 14',
     ' 15', ' 16', ' 17', ' 18', ' 19', ' 20', ' 21',
     ' 22', ' 23', ' 24', ' 25', ' 26', ' 27', ' 28',
     ' 29', ' 30', ' 31', ' X ', ' X ', ' X ', ' X ', ]

"""
ブロックの形状
"""
block_shapes = [
    np.asarray(([
        [0, 0, 1, 1],
        [1, 1, 1, 0]]), dtype=np.uint8),
    np.asarray(([
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 1]]), dtype=np.uint8),
    np.asarray(([
        [1, 0, 1],
        [1, 1, 1]]), dtype=np.uint8),
    np.asarray(([
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 1]]), dtype=np.uint8),
    np.asarray(([
        [0, 0, 1, 0],
        [1, 1, 1, 1]]), dtype=np.uint8),
    np.asarray(([
        [0, 0, 0, 1],
        [1, 1, 1, 1]]), dtype=np.uint8),
    np.asarray(([
        [1, 1, 1],
        [1, 1, 1]]), dtype=np.uint8),
    np.asarray(([
        [1, 1, 0],
        [1, 1, 1]]), dtype=np.uint8),
]


def generate_mat2bin_lut():
    base_bit = 1 << 63
    lut = np.zeros((7, 7), dtype=np.uint64)
    for y in range(7):
        for x in range(7):
            b = 8 * y + x
            lut[y, x] = base_bit >> (63- 8 * y - x) 
    return lut

mat2bin_lut = generate_mat2bin_lut()

def mat2bin(mat):
    assert mat.shape[0] == 7 and mat.shape[1] == 7
    lut = mat2bin_lut.copy()
    lut[mat == 0] = 0
    return np.sum(lut) # as np.unit64

def bin2mat(b):
    mat = np.zeros((7, 7), dtype=np.uint8)
    for y in range(7):
        for x in range(7):
            if (b & mat2bin_lut[y, x]):
                mat[y, x] = 1 
    return mat

def str2matrix(str_mat=None):
    """
    文字列ベースのマトリックスから ndarray を作成

    入力として1次元配列(7*7=49要素)を出力する
    省略した場合は str_matrix が使われる

    出力は shape (7, 7), dtype=np.uint8 の matrix
    入力値が ' X ' の場合は 1, それ以外の場合は 0 に初期化される
    """

    if str_mat is None:
        str_mat = str_matrix

    def func(e): return {True: 1, False: 0}[e == ' X ']
    int_mat = list(map(func, str_mat))
    mat = np.asarray(int_mat, dtype=np.uint8).reshape(-1, 7)

    return mat


def date2strmatrix(month, day):
    """
    日付だけマスクされた ndarray を作成
    このとき指定された月・日も 1 に初期化する
    """

    assert (1 <= month and month <= 12), 'month out of range'
    assert (1 <= day and day <= 31), 'day out of range'

    md = 0 if month >= 7 else -1

    str_mat = [''] * len(str_matrix)
    str_mat[month + md] = ' X '
    str_mat[str_matrix.index('  1') + day - 1] = ' X '

    return str_mat


def new_context(month=None, day=None):
    """
    状態を保存するコンテキスト (dict) を初期化する
    """

    # 事前に同位体を計算
    mat1 = mat2bin(str2matrix(str_matrix))
    mat2 = mat2bin(str2matrix(date2strmatrix(month, day)))
    mat3 = mat1 | mat2

    return {
        'block_bitmap': 0xFF,
        'blocks': [mat1, mat2, ],
        'mat': mat3,
    }


def next_cell(context):
    """
    次に埋めるべき座標を求めます。

    左上〜右上、順番に下まで 0 となっている最初の要素位置を返す
    matrix に 0 （空き）もしくは 1以上（割り当て済み）の値が設定
    されている前提とする
    """

    mat = mat2bin_lut & context['mat']
    pos = np.unravel_index(np.argmin(mat), mat.shape)
    minval = mat[pos[0], pos[1]]

    if minval != 0:
        return None
    return pos


def project_block_prepare(block):
    """
    ブロックの形状から同位体（回転・反転パターン）を洗い出す

    正方形ではないマトリクスの場合 shape が違う2パターンのブロック形状が
    発生するため、ふたつの ndarray を返す
    """
    blocks1 = np.zeros((4, block.shape[0], block.shape[1]), dtype=np.uint8)
    blocks1[0] = block
    blocks1[1] = np.rot90(block, 2)
    blocks1[2] = np.flipud(block)
    blocks1[3] = np.flipud(np.rot90(block, 2))

    blocks2 = np.zeros((4, block.shape[1], block.shape[0]), dtype=np.uint8)
    blocks2[0] = np.rot90(block, 1)
    blocks2[1] = np.rot90(block, 3)
    blocks2[2] = np.flipud(np.rot90(block, 1))
    blocks2[3] = np.flipud(np.rot90(block, 3))

    if blocks1.shape[0] == blocks1.shape[1]:
        blocks = np.concatenate((blocks1, blocks2), axis=0)
        blocks = np.unique(blocks, axis=0)
        return blocks, np.array((0, 1, 1), dtype=np.uint8)
    else:
        blocks1 = np.unique(blocks1, axis=0)
        blocks2 = np.unique(blocks2, axis=0)
        return blocks1, blocks2

def generate_blocks_lut(block_shapes):
    """
    ブロック形状定義リストから、同位体を予め計算したリストを生成する。
    ・np.rot90() などがかなり重いので事前計算しておく
    ・バイナリ化のついでに、想定される回転・反転・位置を予め計算しておく
    """

    def slide(blocks):
        mat = np.zeros((7,7))
        num = blocks.shape[0] * (mat.shape[0] - blocks[0].shape[0] +
                             1) * (mat.shape[1] - blocks[0].shape[1] + 1)
        mat_candidates = np.zeros(
            (num, mat.shape[0], mat.shape[1]), dtype=np.uint8)
        i = 0
        for b in blocks:
            for y in range(mat.shape[0] - b.shape[0] + 1):
                for x in range(mat.shape[1] - b.shape[1] + 1):
                    mat_candidates[i, y: y + b.shape[0], x: x + b.shape[1]] = b
                    i += 1

        return list(map(lambda c: mat2bin(c), mat_candidates[0: i])) # shape=(7, 7) でスライドされたパターンのリスト

    def expand_block(block_shape):
        blocks1, blocks2 = project_block_prepare(block_shape)
        candidates = []
        candidates.extend(slide(blocks1))
        candidates.extend(slide(blocks2))
        mat_candidates = np.asarray(candidates, dtype=np.uint64)
        mat_candidates_uniq = np.unique(mat_candidates, axis=0)

        return mat_candidates_uniq

    packed_blocks = []
    for block_shape in block_shapes:
        packed_blocks.extend([expand_block(block_shape)])
    return packed_blocks


blocks_lut = generate_blocks_lut(block_shapes)

def project_block(context, pos, blocks_uint64):
    """
    blocks に割り当てされたブロックを回転・移動させ、配置できる
    パターン(7, 7)を生成する
    blocks にはひとつのブロックの同位体（回転・反転）のリストを期待する
    座標 pos が埋まるようなパターンのみにフィルタする
    context['mat'] に対する既存ブロックとの衝突をフィルタする
    """

    mat_uint64 = context['mat']

    # pos で指定された座標が埋まっているパターンのみを抽出する。
    bit_pos = mat2bin_lut[pos[0], pos[1]]
    blocks_uint64_masked = blocks_uint64 & bit_pos
    blocks_f1_bool =blocks_uint64_masked != 0
    blocks_f1_uint64 = blocks_uint64[blocks_f1_bool]

    # ブロックが重なりわない ( mat & blocks == 0) 候補のみを抽出する
    blocks_f2_masked = blocks_f1_uint64 & mat_uint64
    blocks_f2_bool =blocks_f2_masked == 0
    blocks_f2_uint64 = blocks_f1_uint64[blocks_f2_bool]

    return blocks_f2_uint64


def step(context):
    """
    コンテキストに対して 1 ステップを実行、するけども再帰で実行するので
    一気に処理される
    """
    # すでに完了しているか?
    if context['block_bitmap'] == 0:
        solutions.append(context['blocks'])
        return

    # 次に埋めるべき場所を決める
    pos = next_cell(context)
    assert (pos is not None), "next_call returned no position"

    mat = context['mat']
    z = 0

    for block_index in range(len(block_shapes)):
        block_bit = (1 << block_index)
        if not (context['block_bitmap'] & block_bit):
            continue

        child_block_bitmap = context['block_bitmap'] ^ block_bit

        block = blocks_lut[block_index]
        projected_blocks_mat = project_block(context, pos, block)

        z += projected_blocks_mat.shape[0]

        # 各パターンを適用したした状態で step() を再帰実行する
        for candidate in projected_blocks_mat:
            new_ctx = {
                'block_bitmap': child_block_bitmap,
                'mat': context['mat'] + candidate,
                'blocks': context['blocks'].copy()
            }
            new_ctx['blocks'].extend([candidate])
            step(new_ctx)

    if (z == 0) and False:
        print("手詰まり depth %d" % len(context['blocks']))
        display(context['blocks'])


def display(blocks):
    """
    ひとつの solution をレンダリングしコンソールに出力する
    """
    for s in render(blocks):
        print(s)


def render(blocks_bin):
    """
    blocks に渡されたパターンのリストからエスケープし〇−件素で結果を
    レンダリングする。
    盤面の文字列については str_matrix を参照する
    盤面の色については escame_sequence_list から、下記が利用される
    escame_sequence_list[0]  ... ブロックが配置されていない部分
    escame_sequence_list[1]  ... (つかってない).
    escame_sequence_list[2~] ... ブロックへの着色のエスケープシーケンス
    """
    escape_sequence_list = [
        '\u001b[97;37m',  # 未割り当て
        '\u001b[97;90m',  # " X "
        '\u001b[97;1m',  # date
        '\u001b[97;41m',
        '\u001b[97;42m',
        '\u001b[97;43m',
        '\u001b[97;44m',
        '\u001b[97;45m',
        '\u001b[97;46m',
        '\u001b[97;47m',
        '\u001b[97;100m',
    ]

    blocks_mat = np.asarray(list(map(lambda b: bin2mat(b), list(blocks_bin))))
    blocks = blocks_mat


    blocks_flatten = np.zeros(
        (len(blocks), blocks[0].reshape(-1).shape[0]), dtype=np.uint8)
    for i in range(len(blocks)):
        blocks_flatten[i, :] = blocks[i].reshape(-1) * (i + 1)

    blocks_marged = np.sum(blocks_flatten, axis=0).reshape(-1, 7)
    texts = []
    texts.extend(str_matrix)

    buf = []
    for y in range(blocks_marged.shape[0]):
        s = ''
        for x in range(blocks_marged.shape[1]):
            t = texts.pop(0)
            c = blocks_marged[y, x]
            s = '%s%s%s\u001b[0m' % (s, escape_sequence_list[c], t)
        buf.append(s)
    return buf


def display_solutions(cols=6):
    """
    solutions コンソールに横 cols 列で須臾力する
    """

    def _flush_solutions(solution_rows):
        """
        solution_rows にある N 個の solution を横並びで出力する
        """
        rendered_solutions = []

        for sol in solution_rows:
            rendered_solutions.append(render(sol))

        while len(rendered_solutions) > 0 and len(rendered_solutions[0]) > 0:
            s = ''
            for rs in rendered_solutions:
                s = '%s   %s' % (s, rs.pop(0))
            print(s)
        print("")

    print("%d solution%s found" %
          (len(solutions), {1: ''}.get(len(solutions), 's')))

    _solutions = solutions.copy()  # pop するので copy
    sols = []
    while len(_solutions):
        sols.append(_solutions.pop(0))
        if cols <= len(sols):
            _flush_solutions(sols)
            sols = []

    if 0 < len(sols):
        _flush_solutions(sols)


if __name__ == '__main__':
    # ターゲット日付
    month, day = None, None
    today = datetime.datetime.fromtimestamp(time.time())
    if len(sys.argv) > 1:
        m = re.match('^(\d+)\/(\d+)$', sys.argv[1])
        if m:
            month, day = int(m.group(1)), int(m.group(2))
        else:
            die("Failed to parse the argument (MM/DD)")

    if month is None:
        month = today.month
    if day is None:
        day = today.day

    print("Seeking for the solutions for date %d/%d\n" % (month, day))

    # solution を探す
    solutions = []
    ctx = new_context(month, day)

    cProfile.run('step(ctx)')

    # solution の整理
    # FIXME: 重複排除は必要か？重複排除前のソートは？
    # FIXME: solutions がグローバル変数渡しになってる

    solutions = np.asarray(solutions, dtype=np.uint64)  # np.unique したいので
    solutions = np.unique(solutions, axis=0)
    solutions = list(solutions)  # pop() したいので

    display_solutions()
