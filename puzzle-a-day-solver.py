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


def masked_matrix(month, day):
    """
    文字列ベースのマトリックスから ndarray を作成
    このとき指定された月・日も 1 に初期化する
    """

    assert (1 <= month and month <= 12), 'month out of range'
    assert (1 <= day and day <= 31), 'day out of range'

    md = 0 if month >= 7 else -1

    str_mat = np.copy(str_matrix)
    str_mat[month + md] = ' X '
    str_mat[list(str_mat).index('  1') + day - 1] = ' X '

    return str2matrix(str_mat)


def new_context(month=None, day=None):
    """
    状態を保存するコンテキスト (dict) を初期化する
    """

    # 事前に同位体を計算
    available_blocks = pack_blocks(block_shapes)
    mat = masked_matrix(month, day)

    return {
        'available_blocks': available_blocks,
        'blocks': [mat],
        'mat': mat,
    }


def next_cell(context):
    """
    次に埋めるべき座標を求めます。

    左上〜右上、順番に下まで 0 となっている最初の要素位置を返す
    matrix に 0 （空き）もしくは 1以上（割り当て済み）の値が設定
    されている前提とする
    """

    mat = context['mat']
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


def pack_blocks(block_shapes):
    """
    ブロック形状定義リストから、同位体を予め計算したリストを生成する。
    np.rot90() などがかなり重いので事前計算しておく

    回転した場合 matrix の shape が2種類発生するので、結果は list にする
    """
    packed_blocks = []
    for block_shape in block_shapes:
        blocks1, blocks2 = project_block_prepare(block_shape)
        packed_block = []
        for n in range(blocks1.shape[0]):
            packed_block.append(blocks1[n])
        for n in range(blocks2.shape[0]):
            packed_block.append(blocks2[n])
        packed_blocks.append(packed_block)
    return packed_blocks


def project_block_internal(context, pos, blocks):
    """
    blocks に割り当てされたブロックを回転・移動させ、配置できる
    パターン(7, 7)を生成する
    blocks にはひとつのブロックの同位体（回転・反転）のリストを期待する
    座標 pos が埋まるようなパターンのみにフィルタする
    context['mat'] に対する既存ブロックとの衝突をフィルタする
    """

    mat = context['mat']

    # のちのフィルタ処理をvectorize したいので
    # ひとつの numpy array に渡された候補を全て渡す
    num = len(blocks) * (mat.shape[0] - blocks[0].shape[0] +
                         1) * (mat.shape[1] - blocks[0].shape[1] + 1)
    mat_candidates = np.zeros(
        (num, mat.shape[0], mat.shape[1]), dtype=np.uint8)
    i = 0
    for b in blocks:
        for y in range(mat.shape[0] - b.shape[0] + 1):
            for x in range(mat.shape[1] - b.shape[1] + 1):
                mat_candidates[i, y: y + b.shape[0], x: x + b.shape[1]] = b
                i += 1

    # pos で指定された座標が埋まっているパターンのみを抽出する。
    mat_pos_flag = mat_candidates[0: i, pos[0], pos[1]] == 1
    mat_candidates_filtered = mat_candidates[mat_pos_flag]

    n = mat_candidates_filtered.shape[0]
    mat_current = np.zeros((n, 7, 7), dtype=np.uint8)
    for i in range(n):
        mat_current[i - 1, :, :] = context['mat']

    # ブロックが重なりあう候補の削除
    mat_candidates_sum = mat_candidates_filtered + mat_current
    mat_candidates_sum = mat_candidates_sum.reshape(
        mat_candidates_sum.shape[0], -1)
    mat_candidates_sum_amax = np.amax(mat_candidates_sum, axis=1)

    # 上記フィルタを通ったものだけ返却
    return mat_candidates_filtered[mat_candidates_sum_amax == 1]


def project_block(context, pos, block):
    """
    block に割り当てされたブロックを回転・移動させ、配置できるパターン(7x7)
    を生成する。
    """
    return project_block_internal(context, pos, block)


def is_done(context):
    """
    全ての枠が埋まっているか評価する
    ・値が絶対に入らないところは str_matrix にて " X " -> 1
    ・日付部分については " X " -> 1
    ・ブロックを配置した部分については 0 -> 1
    このため全ての値が 1 になれば完了
    """
    mat = context['mat']
    return np.min(mat) == 1 and np.max(mat) == 1


def step(context):
    """
    コンテキストに対して 1 ステップを実行、するけども再帰で実行するので
    一気に処理される
    """

    orig_available_blocks = context['available_blocks'].copy()

    # すでに完了しているか?
    if is_done(context):
        solutions.append(context['blocks'].copy())
        # display(context['blocks'])
        return

    assert len(orig_available_blocks) > 0

    # 次に埋めるべき場所を決める
    pos = next_cell(context)
    assert (pos is not None), "next_call returned no position"

    mat = context['mat']

    z = 0
    for block_index in range(len(orig_available_blocks)):
        available_blocks = orig_available_blocks.copy()
        block = available_blocks.pop(block_index)

        projected_blocks_mat = project_block(context, pos, block)
        available_blocks_list = [available_blocks] * \
            int(projected_blocks_mat.shape[0])

        z += projected_blocks_mat.shape[0]

        # note: fit_func() 相当の処理は project 時に行われるようになったので不要

        # 各パターンを適用したした状態で step() を再帰実行する
        for candidate, available_blocks_c in zip(projected_blocks_mat, available_blocks_list):
            # print("available_blocks_c")
            # pprint.pprint(available_blocks_c)
            new_context = context.copy()
            new_context['available_blocks'] = list(available_blocks_c)
            new_context['blocks'] = context['blocks'].copy()
            new_context['blocks'].extend([candidate])
            new_context['mat'] = np.copy(context['mat']) + candidate
            step(new_context)

    if (z == 0) and False:
        print("手詰まり depth %d" % len(context['blocks']))
        display(context['blocks'])


def display(blocks):
    """
    ひとつの solution をレンダリングしコンソールに出力する
    """
    for s in render(blocks):
        print(s)


def render(blocks):
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
        '\u001b[97;31m',
        '\u001b[97;31m',
        '\u001b[97;41m',
        '\u001b[97;42m',
        '\u001b[97;43m',
        '\u001b[97;44m',
        '\u001b[97;45m',
        '\u001b[97;46m',
        '\u001b[97;47m',
        '\u001b[97;41m',  # 使われるのはここまで
        '\u001b[97;42m',
        '\u001b[97;43m',
        '\u001b[97;44m',
        '\u001b[97;45m',
    ]

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

    print("Seeking the solutions for date %d/%d\n" % (month, day))

    # solution を探す
    solutions = []
    ctx = new_context(month, day)

    cProfile.run('step(ctx)')

    # solution の整理
    # FIXME: 重複排除は必要か？重複排除前のソートは？
    # FIXME: solutions がグローバル変数渡しになってる

    solutions = np.asarray(solutions, dtype=np.uint8)  # np.unique したいので
    solutions = np.unique(solutions, axis=0)
    solutions = list(solutions)  # pop() したいので

    display_solutions()
