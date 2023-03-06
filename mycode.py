import collections
import math
import time
from collections import defaultdict

#import torch
import torch


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def build_list(l):
    res = []
    for i in l:
        res.append(ListNode(i))
    for i in range(len(res) - 1):
        res[i].next = res[i+1]
    if len(res) !=0:
        return res[0]

def print_List(head):
    r = []
    while head:
        print(head.val)
        r.append(head.val)
        head = head.next
    print(r)

a = [1,2,3,4,5]
b = [-10,-6,4,6,8]

al = []
bl = []
for i in range(len(a)):
    al.append(ListNode(a[i]))
for i in range(len(a) -1 ):
    al[i].next = al[i+1]
for i in range(len(b)):
    bl.append(ListNode(b[i]))
for i in range(len(b) -1 ):
    bl[i].next = bl[i+1]





res = []
path = []


def dfs(root, target):
    if not root:
        return []
    print(target, path, sum(path), len(path))
    if not root.left and not root.right:
        # if not root:
        if target == root.val:
            # path.append(root.val)
            res.append(path[:] + [root.val])
            # return res
        else:
            print('else', path)
            # path.pop(-1)

    path.append(root.val)
    print(path)
    if root.left:
        dfs(root.left, target - root.val)
        path.pop(-1)
    if root.right:
        dfs(root.right, target - root.val)
        path.pop(-1)


a = [5,4,3,2,1]
#a = [1,4,3,5,2]
def quick_sort(l,r):
    print(l,r)
    i,j = l, r
    p = a[0]
    if l>=r:
        return
    while i < j:
        while a[l] < a[j] and i<j:
            j -= 1

        a[i] = a[j]
        while a[l] > a[i] and i<j:
            i += 1
        a[j] = a[i]
    a[j] = p
    #     a[i],a[j] = a[j], a[i]
    # a[l], a[i] = a[i], a[l]

    quick_sort(l, i-1)
    quick_sort(i+1, l)


#print('-------')

def strToInt(str: str) -> int:
    str = str.strip(' ')
    states = [
        {'sign':1, 'digit':1},    # sign
        {'digit':1}     # digit
        #{'other':2}
    ]
    #print(1)
    p = 0
    res = ''
    #for i in range(len(str)):
    for i in str:
        if i in '+-':
            t = 'sign'
            #res += i
        elif  '0' <= i <='9':
            t = 'digit'
            #res += i
        else:
            t = '?'
        #print(i, res, p)
        if t not in states[p]:
            #print('not in ')
            if len(res) > 1:
                break
                #return int(res)
            elif len(res) ==1:
                if res[0] in '+-':
                    return 0
                else:
                    #return int(res)
                    break
            else:
                return 0
        res += i
        p = states[p][t]


    #print(res)
    if len(res) > 1:
        if  -pow(2,31)<= int(res) <=pow(2,31) -1:
            #print('res->',res)
            return int(res)
        else:
            if int(res) >0:
                return pow(2,31) -1
            else:
                return -pow(2,31)
    else:
        #print('0-->',0)
        return 0 if res in '+-' else int(res)

from enum import Enum
class Solution:
    def isNumber(self, s: str) -> bool:
        State = Enum("State", [
            "STATE_INITIAL",
            "STATE_INT_SIGN",
            "STATE_INTEGER",
            "STATE_POINT",
            "STATE_POINT_WITHOUT_INT",
            "STATE_FRACTION",
            "STATE_EXP",
            "STATE_EXP_SIGN",
            "STATE_EXP_NUMBER",
            "STATE_END"
        ])

        def toChartype(ch: str):
            if ch.isdigit():
                return 'CHAR_NUMBER'
            elif ch.lower() == "e":
                return 'CHAR_EXP'
            elif ch == ".":
                return 'CHAR_POINT'
            elif ch == "+" or ch == "-":
                return 'CHAR_SIGN'
            elif ch == " ":
                return 'CHAR_SPACE'
            else:
                return 'CHAR_ILLEGAL'

        transfer = {
            'STATE_INITIAL': {
                'CHAR_SPACE': 'STATE_INITIAL',
                'CHAR_NUMBER': 'STATE_INTEGER',
                'CHAR_POINT': 'STATE_POINT_WITHOUT_INT',
                'CHAR_SIGN': 'STATE_INT_SIGN'
            },
            'STATE_INT_SIGN': {
                'CHAR_NUMBER': 'STATE_INTEGER',
                'CHAR_POINT': 'STATE_POINT_WITHOUT_INT'
            },
            'STATE_INTEGER': {
                'CHAR_NUMBER': 'STATE_INTEGER',
                'CHAR_EXP': 'STATE_EXP',
                'CHAR_POINT': 'STATE_POINT',
                'CHAR_SPACE': 'STATE_END'
            },
            'STATE_POINT': {
                'CHAR_NUMBER': 'STATE_FRACTION',
                'CHAR_EXP': 'STATE_EXP',
                'CHAR_SPACE': 'STATE_END'
            },
            'STATE_POINT_WITHOUT_INT': {
                'CHAR_NUMBER': 'STATE_FRACTION'
            },
            'STATE_FRACTION': {
                'CHAR_NUMBER': 'STATE_FRACTION',
                'CHAR_EXP': 'STATE_EXP',
                'CHAR_SPACE': 'STATE_END'
            },
            'STATE_EXP': {
                'CHAR_NUMBER': 'STATE_EXP_NUMBER',
                'CHAR_SIGN': 'STATE_EXP_SIGN'
            },
            'STATE_EXP_SIGN': {
                'CHAR_NUMBER': 'STATE_EXP_NUMBER'
            },
            'STATE_EXP_NUMBER': {
                'CHAR_NUMBER': 'STATE_EXP_NUMBER',
                'CHAR_SPACE': 'STATE_END'
            },
            'STATE_END': {
                'CHAR_SPACE': 'STATE_END'
            },
        }

        st = 'STATE_INITIAL'
        for ch in s:
            typ = toChartype(ch)
            if typ not in transfer[st]:
                return False
            st = transfer[st][typ]

        return st in ['STATE_INTEGER', 'STATE_POINT', 'STATE_FRACTION', 'STATE_EXP_NUMBER',
                      'STATE_END']


def strToInt(str: str):
    transfer = {
        'state_start':{
            'char_space':'state_start','char_sign':'state_sign',
            'char_number':'state_number','other':'state_break'
                        },
        'state_sign':{
            'char_number':'state_number','other':'state_break'
        },
        'state_number':{
            'char_number':'state_number','other':'state_break'
        },
        'state_break':{
            'char_break':'state_break'
        }
    }

    state = 'state_start'
    c = 0
    res = ''
    for i in str:
        print(transfer[state])

        if i ==' ':
            c = 'char_space'
        elif i in '+-':
            c = 'char_sign'
            if len(res) ==0:
                res += i
        elif i.isdigit():
            c = 'char_number'
            res += i
        else:
            c = 'other'
        print(c)
        print(res)
        if c not in transfer[state]:
            print('break')
            break

        state = transfer[state][c]
        if state == 'state_break':
            break

    if len(res) > 1 or (len(res) == 1 and res[0] not in '+-'):
        if -pow(2,31)<= int(res) <= pow(2,31) -1:
            return int(res)
        else:
            if int(res)>0:
                return pow(2,31)-1
            else:
                return -pow(2,31)
    else:
        return 0


#
# def isMatch(s: str, p: str) -> bool:
#     print('hello world')
#
#     states = {
#         'state_start': {
#             'point': 'state_point',
#             'char': 'state_char'
#         },
#         'state_point':{
#             'point':'state_point',
#             'char':'state_char',
#             'star':'state_star'
#         },
#         'state_char':
#             {
#                 'point':'state_point',
#                 'char':'state_char',
#                 'star':'state_star'
#             },
#         'state_star':{
#             'point':'state_point',
#             'char':'state_char',
#             'star':'state_star'
#         }
#     }
#
#
#     sindex = 0
#     pindex = 0
#     state = 'state_start'
#     #while index < len(s):
#     last_char = ''
#     #for i in range(len(p)):
#     char = 'inin'
#     while pindex < len(p) and sindex < len(s):
#         #print(sindex)
#         print(char, state)
#         if p[pindex] == '.':
#             char = 'point'
#             sindex += 1
#             pindex += 1
#         elif p[pindex] == '*':
#             char = 'star'
#             #index += 1
#             #if s[index]
#         elif  'a' <= p[pindex] <= 'z':
#             char = 'char'
#             if  s[sindex] == p[pindex]:
#                 sindex += 1
#                 pindex += 1
#                 # print('sindex',sindex)
#                 # print('相同',s[sindex])
#             else:
#                 print('*判断break')
#                 print(s[sindex], p[pindex])
#                 break
#         else:
#             char = '?'
#         if char not in states[state]:
#
#             print('异常状态',char, state)
#             #return False
#             break
#         if pindex >= len(p) or sindex >= len(s):
#             break
#         if p[pindex] == '*':
#             if p[pindex - 1] == '.':
#                 if pindex < len(p)-1 and 'a'<=p[pindex + 1]<='z':
#                     pass
#                 else:
#                     sindex += 1
#             else:
#                 if s[sindex] == s[sindex - 1]:
#                     sindex += 1
#
#                     print('多次',p[pindex - 1])
#                     print(p[pindex:], s[sindex:])
#                 else:
#                     #sindex += 1
#                     pindex += 1
#                     print('零次',p[pindex - 1])
#         state = states[state][char]
#
#     #print(sindex,s[sindex], pindex, p[pindex])
#     print(sindex, pindex)
#     if sindex == len(s) and pindex == len(p):
#         return True
#     else:
#         print('最后false')
#         return False


class Automata:
    def __init__(self, p: str) -> None:
        self.transit_tokens = []
        self.loop_tokens = {}
        i = 0

        while i + 1 < len(p):
            if p[i + 1] == '*':
                cur_state = len(self.transit_tokens)
                if cur_state in self.loop_tokens:
                    self.loop_tokens[cur_state] += p[i]
                else:
                    self.loop_tokens[cur_state] = p[i]
                self.transit_tokens.append('')
                i += 2
            else:
                self.transit_tokens.append(p[i])
                i += 1

        if i < len(p): self.transit_tokens.append(p[i])
        self.state_cnt = len(self.transit_tokens)
        self.compute_epsilon_closure()
        self.states = self.epsilon_closure[0]

    def compute_epsilon_closure(self) -> None:
        """
        Note that in this regular expression case the direction of \epsilon arrow will always
        point from left to next right node.
        Hence we simply backward iterate the array with step=1 to get closure.
        """
        self.epsilon_closure = [{state} for state in range(self.state_cnt + 1)]
        for i in reversed(range(self.state_cnt)):
            if self.transit_tokens[i] == '':
                self.epsilon_closure[i] = self.epsilon_closure[i].union(self.epsilon_closure[i + 1])

        # print(self.epsilon_closure)

    def run(self, symbol: str) :
        new_states = set()

        for state in self.states:
            if state < self.state_cnt and (symbol in self.transit_tokens[state] or '.' in self.transit_tokens[state]):
                new_states.add(state + 1)
            if state in self.loop_tokens and (symbol in self.loop_tokens[state] or '.' in self.loop_tokens[state]):
                new_states.add(state)

        new_states = new_states.union(*[self.epsilon_closure[new_state] for new_state in new_states])

        # print(self.states, symbol, new_states)
        self.states = new_states
        print(self.states)
        return new_states


class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        if len(s) == 0 and len(p) == 0: return True
        if len(s) != 0 and len(p) == 0: return False

        dfa = Automata(p)
        for c in s:
            states = dfa.run(c)

            if len(states) == 0: return False
        return (dfa.state_cnt in dfa.states)


#   dp正则
import numpy as np
def isMatch(s: str, p: str) -> bool:
    # m, n = len(s), len(p)
    # dp = [[False] * (n + 1) for _ in range(m + 1)]
    #
    # # 初始化
    # dp[0][0] = True
    # for j in range(1, n + 1):
    #     if p[j - 1] == '*':
    #         dp[0][j] = dp[0][j - 2]
    #
    # print(np.array(dp))
    # # 状态更新
    # for i in range(1, m + 1):
    #     for j in range(1, n + 1):
    #         if s[i - 1] == p[j - 1] or p[j - 1] == '.':
    #             dp[i][j] = dp[i - 1][j - 1]
    #         elif p[j - 1] == '*':  # 【题目保证'*'号不会是第一个字符，所以此处有j>=2】
    #             if s[i - 1] != p[j - 2] and p[j - 2] != '.':
    #                 dp[i][j] = dp[i][j - 2]
    #             else:
    #                 dp[i][j] = dp[i][j - 2] | dp[i - 1][j]
    #
    # print(np.array(dp))
    # return dp[-1][-1]


    dp = [[0]*(len(p)+1) for _ in range(len(s)+1)]
    #print(np.array(dp).shape , len(s), len(p))
    dp[0][0] = True
    for j in range(1, len(p)+1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]
        #dp[0][j] = dp[0][j - 2] and p[j - 1] == '*'





    print(np.array(dp))




    # for i in range(1, len(s) + 1):
    #     for j in range(1, len(p) + 1):
    #         if s[i - 1] == p[j - 1] or p[j - 1] == '.':
    #             dp[i][j] = dp[i - 1][j - 1]
    #         elif p[j - 1] == '*':  # 【题目保证'*'号不会是第一个字符，所以此处有j>=2】
    #             if s[i - 1] != p[j - 2] and p[j - 2] != '.':
    #                 dp[i][j] = dp[i][j - 2]
    #             else:
    #                 dp[i][j] = dp[i][j - 2] | dp[i - 1][j]

    for i in range(1,len(s)+1):
        for j in range(1,len(p)+1):
            #print(i, j)
            if s[i - 1] == p[j - 1] or p[j - 1] == '.':
                 dp[i][j] = dp[i - 1][j - 1]
            # elif
            #     dp[i][j] = dp[i-1][j-1]
            elif p[j - 1] == '*':  # 【题目保证'*'号不会是第一个字符，所以此处有j>=2】
                if s[i - 1] == p[j - 2]:
                    dp[i][j] = dp[i - 1][j] | dp[i][j-2]
                elif p[j - 2] == '.':
                    dp[i][j] = dp[i][j-2] | dp[i-1][j]
                    # if dp[i-1][j]:
                    #     dp[i][j] = dp[i-1][j]
                    # else:
                    #     dp[i][j] = dp[i][j-2]
                else:
                    dp[i][j] = dp[i][j - 2]




    print(np.array(dp))
    if dp[-1][-1] == 1:
        return True
    else:
        return False

s = "aaa"
#p = "ab*a*c*a"
#print(Solution().isMatch(s, p))
#print(isMatch(s,p))

def permutation(s: str) :
    res = []

    def backtracking(s, path):
        #print(path)
        if len(s) == 0:
            res.append(path[:])
            return

        record = set()
        for i in range(len(s)):

            if s[i] in record:
                continue
            record.add(s[i])
            #print(path)
            backtracking(s[:i]+s[i+1:], path + s[i])


    backtracking(s, "")
    return res

    # res = []
    #
    # def backtrack(s, path):
    #     if not s:
    #         res.append(path)
    #     seen = set()
    #     for i in range(len(s)):
    #         if s[i] in seen: continue
    #         seen.add(s[i])
    #         backtrack(s[:i]+s[i+1:], path + s[i])
    #
    # backtrack(s, "")
    # return res
#print(permutation('aab'))

# 39
nums = [1, 2, 3, 2, 2, 2, 5, 4, 2]
def majorityElement(nums):
    res = 0
    record = 0
    for i in range(len(nums)):
        if res ==0:
            record = nums[i]
        if nums[i] == record:
            res += 1
        else:
            res -= 1

        print(f'当前{nums[i]},分数{res}, 标签{record}')
    return record

#print(majorityElement(nums))
def verifyPostorder(self, postorder) -> bool:


    def bfs(left, right):
        if right - left <=1:
            return True

        for i in range(left, right):
            if postorder[i] >= postorder[right]:

                middle = i

        for i in range(middle, right):
            if postorder[i] > right:
                return False

        return postorder(left, middle) and postorder(middle, right)


import sys
class Solution:
    def maxSubArray(self, nums) -> int:
        return self.helper(nums, 0, len(nums) - 1)
    def helper(self, nums, l, r):
        if l > r:
            return -sys.maxsize
        mid = (l + r) // 2
        left = self.helper(nums, l, mid - 1)
        right = self.helper(nums, mid + 1, r)
        left_suffix_max_sum = right_prefix_max_sum = 0
        sum = 0
        for i in reversed(range(l, mid)):
            sum += nums[i]
            left_suffix_max_sum = max(left_suffix_max_sum, sum)
        sum = 0
        for i in range(mid + 1, r + 1):
            sum += nums[i]
            right_prefix_max_sum = max(right_prefix_max_sum, sum)
        cross_max_sum = left_suffix_max_sum + right_prefix_max_sum + nums[mid]
        print(f'cross {cross_max_sum}, left {left_suffix_max_sum} right {right_prefix_max_sum} nums[mid] {nums[mid]}, mid {mid} l {l} r {r}')
        return max(cross_max_sum, left, right)


nums = [-2,1,-3,4,-1,2,1,-5,4]
#f = Solution().maxSubArray(nums)

#   48
s = 'dvdf'
s = "pwwkew"
s = "abcabcbb"
def lengthOfLongestSubstring(s: str) -> int:
    slow = 0
    fast = 0
    seen = set()
    length = 0
    #for i in range(len(s)):
    while fast < len(s) and slow < len(s):
        # if s[fast] not in seen:
        #     fast += 1
        #     seen.add(s[fast])
        # else:
        #     slow += 1
        #
        # length = max(length, fast - slow)
        # print(s[slow:fast+1], seen, slow, fast)
        seen = set()
        if s[fast] not in seen:
            fast += 1
            seen.add(s[fast])
        else:
            slow += 1
        print(s[slow:fast])
    print(length)
#lengthOfLongestSubstring(s)


#   63
prices= [7,1,5,3,6,4]

#prices = []

#prices = [1,2,4,2,5,7,2,4,9,0,9]


def maxProfit(prices) -> int:
    #dp = [0] *len(prices)
    dp=0
    minvalue = float('inf')

    for i in range(len(prices)):
        minvalue = min(minvalue, prices[i])
        dp = max(dp, prices[i] - minvalue)


    print(dp)

#maxProfit(prices)

p = [0.16667,0.16667,0.16667,0.16667,0.16667,0.16667]

n = 2
s = []

import numpy
def dicesProbability(n: int):
    dp = [[0] *(6*n+1) for _ in range(n)]
    #     #dp[:6] = [0.16667,0.16667,0.16667,0.16667,0.16667,0.16667]
    #     for i in range(6):
    #         dp[0][i+1] =  round(1.0 /6,5)
    #
    #     print(np.array(dp))
    #     for i in range(1,n):    #   骰子索引
    #         for j in range(i+1,6*(i+1)+1):  #   点数索引
    #             # if j < n:
    #             #     continue
    #             for k in range(6):  #   偏差
    #                 #print(i,j,k)
    #                 #if j-k-1>0:
    #                 if j-k-1 <= 0 or j - k - 1 > i * 6:
    #                     continue
    #                 #print(f'更新点数和为{j}, 第{i+1}个骰子点数为{k+1}', j - k - 1 , k+1)
    #                 #print(dp[i-1][j - k - 1], dp[0][k+1], i,j)
    #                 #dp[i][j] += round(dp[i-1][j - k-1] * dp[0][k+1],5)
    #                 dp[i][j] += dp[i-1][j - k - 1] * dp[0][k + 1]
    #                 #print(dp[i])
    #                 #dp[i][j] = round(dp[i][j], 5)
    #     # print(np.array(dp).shape)
    #     #
    #     #
    #     # print(np.array(dp))
#dicesProbability(3)

#   66

a = [1,2,3,4,5]
a = []
def constructArr(a):

    b = [1]*(len(a))
    tmp = 1
    for i in range(1,len(a)):
        b[i] = b[i-1]*a[i-1]

    for i in range(len(a)-1, 0,-1):
        tmp *= a[i]
        b[i-1] = b[i-1]*tmp

    print(b)
constructArr(a)

#   61

nums = [1,2,3,4,5]
#nums = [0,0,1,2,5]
nums = [0,0,8,5,4]
#nums = [0,0,2,2,5]
#nums = [8,2,9,7,10]
def isStraight(nums) -> bool:
    nums = sorted(nums)
    zeros = 0
    print(nums)
    for i in range(len(nums)-1):
        if nums[i] == 0:
            zeros += 1
            continue
        dis = nums[i+1] - nums[i]
        if dis >1:
            if zeros == 0:
                return False
            else:
                print('start',dis, zeros)
                while dis > 1:
                    dis -= 1
                    zeros -= 1
                print('end',dis,zeros)
        elif dis == 0:
            return False

    if zeros < 0:
        return False
    return True

#print(isStraight(nums))

# 65

a ,b = 1,3


def add(a: int, b: int) -> int:

    up =1
    #while up:
    while b:
        #print('ab',a,b)
        print(1)
        # local = a ^ b   #   本位
        # up = a & b <<1  #   进位

        c = a&b
        d = a ^b
        b = c<<1
        a = d

        #print('cd',c,d)
        #
        # a = local
        # b = up
        #print(a,b,c,d)

    #print(r)
    return

#print(add(a,b))
#   62
n = 88
m = 10
def lastRemaining(n: int, m: int) -> int:

    # l = [i for i in range(n)]
    # index = 1
    # del_index = 0
    # while len(l) > 1:
    #     print(f'面上指针{index},删除指针{del_index}')
    #     if index != m:
    #         index += 1
    #         del_index = (del_index + 1) % len(l)
    #     else:
    #         print(f'弹出{l[del_index]}')
    #         l.pop(del_index)
    #         index = 1
    #         #del_index = del_index + 1
    #     #print(l)
    #     #print(index, l[index-1])
    #
    #     print(l)
    # return l[-1]
    l = [i for i in range(n)]
    def dfs(l):
        if len(l) == 1:
            return l[0]
            # if m%2 ==0:
            #     return l[-1]
            # else:
            #     return l[0]
        else:
            if m % len(l) !=0:
                print(m%len(l))
                l = l[m % len(l):] + l[:m % len(l)-1]
            else:
                l = l[:len(l) -1]
            print(l)
            res = dfs(l)
        return res
    r = dfs(l)
    print(r)



#lastRemaining(n,m)

class MaxQueue:
    def __init__(self):
        from collections import defaultdict
        self.stack = []
        self.monotonous = []

    def max_value(self) -> int:
        if len(self.stack) == 0:
            return -1
        else:
            print('当前',self.monotonous)
            return self.monotonous[0]

    def push_back(self, value: int) -> None:
        self.stack.append(value)
        tmp_stack = []
        while len(self.monotonous)>0 and value > self.monotonous[-1]:
            tmp_stack.append(self.monotonous.pop())
        #print(tmp_stack)
        self.monotonous.append(value)

        print('单调栈',self.monotonous)
    def pop_front(self) -> int:
        if len(self.stack) == 0:
            return -1
        else:
            pop_item = self.stack.pop(0)
            if pop_item == self.monotonous[0]:
                self.monotonous.pop(0)

            print(f'pop {pop_item}后单调栈',self.monotonous)
            return pop_item


#   59
nums = [1,3,-1,-3,5,3,6,7]
k = 3
nums = [1,-1]
k = 1

def maxSlidingWindow(nums, k):
    s = []
    res = []

    for i in range(k):
        while s and nums[s[-1]] < nums[i]:
            #print(f'{s[-1]} < {nums[i]}')
            s.pop()
        s.append(i)
        print(nums[i])

    res.append(nums[s[0]])
    print([nums[i] for i in s], res)
    print('---')
    for i in range(k, len(nums)):

        if s and i > s[0] + k -1:
            #print(f'到了{nums[i]}pop {nums[s[0]]}')
            s.pop(0)
        while s and nums[s[-1]] < nums[i]:
            print(f'{[nums[i] for i in s]},{nums[s[-1]]}  < {nums[i]}')
            s.pop()
        s.append(i)
        res.append(nums[s[0]])
        #print(nums[i-k+1:i+1], [nums[i] for i in s])

    print(res)
#maxSlidingWindow(nums, k)

#   58

s = "abcdefg"
k = 2

# s = "lrloseumgh"
# k = 6
def reverseLeftWords(s, n):
    s = [i for i in s]
    siz = len(s)
    for i in range(n):
        s.append(s[i])

    for i in range(siz):
        s[i] = s[i+n]

    s = ''.join(s[:siz])
    print(s)
#reverseLeftWords(s, k)

# 58-1

#  输入: "the sky is blue"
# 输出: "blue is sky the"
s =  "the sky is blue"
s = "  hello world!  "
def reverseWords(s):
    s = s.split(' ')
    s = [i for i in s if i]
    print(' '.join(s[::-1]))
    return s[::-1]
#reverseWords(s)
n = 10
def nthUglyNumber(n: int) -> int:
    dp = [0,1,2,3,4,5] + [0] *(n-5)
    for i in range(n-5 + 1):
        for j in dp:
            for k in dp:
                if j*k not in dp:

                    dp[j*k] = j*k
    print(dp)

#nthUglyNumber(n)

#   48
s = "abcabcbb"
#s = "pwwkew"
s = "anviaj"
#s = "abba"
def lengthOfLongestSubstring(s):

    if len(s) == 0:
        return 0
    dp = [0] * len(s)
    seen = {}
    res = 0
    head = 0
    for i in range(len(s)):
        # print(i, s[i])

        if s[i] in seen:
            head = max(seen[s[i]] + 1, head)
            print('head-->',head)
        seen[s[i]] = i
        print(f'max {res}, {i-head}', f'i = {i} head = {head}', seen)
        res = max(res, i - head )
    print(res)
    return res
    #print(dp)
#lengthOfLongestSubstring(s)
#   47
grid =  [
 [1,3,1],
 [1,5,1],
 [4,2,1]
 ]


#grid = []
def maxValue(grid):

    if len(grid) ==0:
        return 0

    path = []
    res = []
    def backtracking(row, col):
        if row == len(grid)-1 and col == len(grid[0])-1:
            #res.append(path[:] + [grid[-1][-1]])
            res.append(sum(path[:] + [grid[-1][-1]]))
            print(res)
        path.append(grid[row][col])
        if row < len(grid) -1:
            backtracking(row+1, col)
        if col < len(grid[0]) -1:
            backtracking(row, col+1)
        path.pop()

    backtracking(0,0)

    #print(res)
    return max(res)
#maxValue(grid)

#   46
num = 12258
#num = 26
num = 1
def translateNum(num):

    dp = [0]*len(str(num))
    dp[0] = 1
    if len(str(num)) >2:
        if '0'<=str(num)[0] + str(num)[1] <='25' and str(num)[0] !=0:
            dp[1] = 2
        else:
            dp[1] = 1
    #print(str(num))
    for i in range(2,len(str(num))):
        #print(str(num)[i-1] ,str(num)[i])
        if '0'<=str(num)[i-1] + str(nums)[i] <='25' and str(num)[i-1] !='0':
            dp[i] = dp[i-1] + dp[i-2]
        else:
            dp[i] = dp[i-1]
    print(dp)
    return dp[-1]

#translateNum(num)

#   45
nums = [3,30,34,5,9]

#nums = ['1','2']
#nums = [128,12]
nums = [1,2,3,1]

nums = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
nums = [0]*100
def minNumber(nums):
    nums = sorted(nums)
    nums = [f'{i}' for i in nums]
    res = []
    path = ''
    seen = []
    #length = length(''.join(nums))
    #print(''.join(nums))
    #print(len(''.join(nums)))

    def backtracking(path):

        if len(path) == len(str(''.join(nums))) and path not in res:

            res.append(path[:])

        for index in range(len(nums)):
            if index in seen:
                continue
            #path.append(nums[index])

            if len(path) != 0 and res and path + str(nums[index])  > res[-1][:len(path+str(nums[index]))]:
                print(path, res[-1][:len(path)])
                continue
            if nums[index] == nums[index-1] and index-1 in seen:
                continue
            print(len(path),len(res))
            path += str(nums[index])

            seen.append(index)
            backtracking(path)

            #path.pop()
            #print('before',path)
            path = path[:len(path) - len(str(nums[index]))]
            #print(path)
            seen.pop()
    backtracking(path)

    print(np.array(res))
#minNumber(nums)
#   05
s = "We are happy."

s =' '
def replaceSpace(s):
    spaceNums = s.count(' ')
    s = list(s)
    i = len(s) -1
    s += ' '*2*spaceNums
    j = len(s) - 1
    while j > 0:
        print(i,j, s[i], s[j])
        if s[i] != ' ':
            s[j] = s[i]
            i -=1
            j -=1
        else:
            s[j-2:j+1] = '%20'
            i -=1
            j -=3
    print(''.join(s))
    return ''.join(s)
        #print(''.join(s))
    #print(s)

#replaceSpace(s)
# 57
target = 9
target = 10000
def findContinuousSequence(target):
    res = []
    path = []

    def backtracking(start, target):
        #if sum(path) == target:
        print(target, path, res, start)
        if target == 0:
            res.append(path[:])

        for index in range(start, target+1):

            if len(path)!=0 and index - path[-1] > 1:
                return
                #continue
            if index == target and len(path) == 0:
                continue
            path.append(index)

            print(index, path, sum(path))
            backtracking(index + 1, target - index)
            path.pop()

    backtracking(1,target)
    print(res)

#findContinuousSequence(target)

#   10
n = 45
def fib(n):
    dp = [0]*(n+1)
    dp[0] = 0
    dp[1] = 1
    for i in range(2,n+1):
        dp[i] = dp[i-1]+dp[i-2]
    print(dp)
    for i,j in enumerate(dp):
        print(i,j % 1000000007)


    return dp[-1]
#fib(n)

#   10-2
n = 1
def numWays(n):


    f = [0]*(n)

    #f[2] = 1
    def w(n):
        #print(f)
        if n==0:
            return 0
        elif n == 1:
            return 1
        if f[n-1]!=0:
            return f[n-1]
        else:
            f[n-1] = w(n-1) + w(n -2)
            return f[n-1]

    w(n)
    print(f)
    #return f[n-1]
#numWays(n)

#   68
def lowestCommonAncestor(root, p,q):

    if root.val == p.val or root.val == q.val:
        return root

    if p.val + q.val > 2*root.val:
        return root
    else:
        if p.val < root.val:
            return lowestCommonAncestor(root.left, p, q)
        else:
            return lowestCommonAncestor(root.right, p, q)

#   68-2
def lowestCommonAncestor(root, p,q):

    if not root:
        return root

    if root == p or root == q:
        return root

    left = self.lowestCommonAncestor(root.left, p, q)
    right = self.lowestCommonAncestor(root.right, p, q)

    if not left and not right:
        return
    if left and not right:
        return left
    if right and not left:
        return right
    if left and right:
        return root


#   26
head = build_list([1,2,3,4])
def reorderList(head):

    h = head
    l = []
    while h:
        l.append(h)
        print(h.val)
        h = h.next

    i =0
    #for i in range(len(l)-1):
    while i < len(l) - 1- i:
        print(f'i -> {i} , next -> {len(l) - 1 -i} , i+1 -> {i+1} ')
        l[i].next = l[len(l) - 1 - i]
        #if i+1 < len(l) -i-1:
        l[len(l) - 1 - i].next = l[i+1]
        # else:
        #     print('iiiii',i)
        #     l[len(l)-1-i].next = None


        i +=1

    l[len(l) //2].next = None
    #print('!!!!!!!!!!!!!!')
    head = l[0]
    #print(l[-2].next.val)
    while head:

        print(head.val)
        head = head.next
#reorderList(head)

#   21
head = [1,2,3,4,5]
n = 2
head = [1,2,3]
n = 1
head = build_list(head)
def removeNthFromEnd(head: ListNode, n: int) -> ListNode:
    newHead = ListNode(0)
    newHead.next = head
    head = newHead

    if not head.next:
        return
    slow = fast = head

    for _ in range(n):
        fast = fast.next

    while fast.next:
        fast = fast.next
        slow = slow.next
    print('f',fast.val, slow.val)

    #if slow.next and slow.next.next:
    tmp = slow.next.next
    slow.next.next = None
    slow.next = tmp

    print('-----')
    head = head.next
    while head:
        print(head.val)
        head = head.next
#removeNthFromEnd(head, n)

#   quick_sort

def quick_sort(mylist):

    left = 1
    right = len(mylist) - 1
    while left < right:
        print(left, right)
        while mylist[left] > mylist[0]:
            left += 1
        while mylist[right] < mylist[0]:
            right -= 1

        mylist[left], mylist[right] = mylist[right], mylist[left]

        mylist[left], mylist[0] = mylist[0], mylist[left]

    print(mylist)

def merge_sort(mylist):
    # def divide(mylist):
    #     left = 0
    #     right = len(mylist)
    #     if left >= right:
    #         return 0
    #     else:
    #         return (left + right) //2
    #
    #
    # def merge(leftList, rightList):
    #     res = []
    #     i = 0
    #     j = 0
    #     while i < len(leftList) and j < len(rightList):
    #         if leftList[i] <= rightList[j]:
    #             #print(1)
    #             res.append(leftList[i])
    #             i += 1
    #         else:
    #             #print(2)
    #             res.append(rightList[j])
    #             j += 1
    #
    #     while i < len(leftList):
    #         #print(3)
    #         res.append(leftList[i])
    #         i += 1
    #     while j < len(rightList):
    #         #print(4)
    #         res.append(rightList[j])
    #         j += 1
    #
    #     return res
    #
    # #print(mylist)
    # if len(mylist) < 2:
    #     return mylist
    # left = 0
    # #mid = (left + len(mylist))//2
    # mid = divide(mylist)
    # print(mylist[:mid], mylist[mid:])
    # res = merge(merge_sort(mylist[:mid]), merge_sort(mylist[mid:]))
    # print('res->',res)
    # return res

    #   迭代
    # def merge(arr, left, mid, right):
    #     leftList = arr[left:mid]
    #     rightList = arr[mid:right]
    #     res = []
    #     i = 0
    #     j = 0
    #     while i < len(leftList) and j < len(rightList):
    #         if leftList[i] <= rightList[j]:
    #             #print(1)
    #             res.append(leftList[i])
    #             i += 1
    #         else:
    #             #print(2)
    #             res.append(rightList[j])
    #             j += 1
    #
    #     while i < len(leftList):
    #         #print(3)
    #         res.append(leftList[i])
    #         i += 1
    #     while j < len(rightList):
    #         #print(4)
    #         res.append(rightList[j])
    #         j += 1
    #
    #     #mylist = res
    #     print('res->',res)
    #     return res
    def merge(arr, left, mid, right):
        a = [0]*(right - left + 1)
        i = left
        j = mid + 1
        k = 0
        while i <= mid and j <= right:
            if arr[i] < arr[j]:
                a[k] = arr[i]
                k += 1
                i += 1
            else:
                a[k] = arr[j]
                k += 1
                j += 1
        while i<=mid:
            a[k] = arr[i]
            k += 1
            i += 1
        while j <= right:
            a[k] = arr[j]
            k += 1
            j += 1

        for i in range(k):
            arr[left] = a[i]
            left += 1


    n = len(mylist)

    for i in range(1, n):
        left = 0
        mid = left + i -1
        right = mid + i
        while right < n:

            print('mylist ',mylist, f'left -> {left}, mid -> {mid}, right -> {right}',mylist[left:mid], mylist[mid:right])
            merge(mylist, left, mid, right)
            left = right + 1
            mid = left + i -1
            right = mid + i

        if left < n and mid < n:
            print('mylist ',mylist, f'left -> {left}, mid -> {mid}, right -> {right}',mylist[left:mid], mylist[mid:right])
            merge(mylist, left, mid, n-1)

    print(mylist)
    return mylist


#head = build_list([4,2,1,4,-1])
head = build_list([-1,5,3,4,0])
#head = build_list([])
# while head:
#     print(head.val)
#     head = head.next
def sortList(head: ListNode):

    if not head.next:
        return head

    def merge(left, right):
        dummy = ListNode(-1)
        newHead = dummy
        while left and right:

            if left.val <= right.val:
                newHead.next = left
                left = left.next

            else:
                newHead.next = right
                right = right.next

            newHead = newHead.next

        while left:

            newHead.next = left
            left = left.next
            newHead = newHead.next
        while right:

            newHead.next = right
            right = right.next
            newHead = newHead.next


        return dummy.next

    def divid(head):

        dummy = ListNode(-1)
        dummy.next = head
        slow = fast = dummy

        t = dummy
        while t:

            t = t.next


        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next


        right = slow.next
        slow.next = None
        return right

    left = head
    right = divid(head)

    #print('left.val',left.val, 'right.val', right.val)
    r = merge(sortList(left), sortList(right))

    return r

#   25
def reverList(head):
    dummy = ListNode(-1)
    dummy.next = head
    newHead = head
    head = dummy
    while dummy:
        #print(dummy.val)
        dummy = dummy.next
    #time.sleep(5)
    tmp = head.next.next
    nex = head.next
    while tmp:


        #print(head.val, nex.val, tmp.val)
        #time.sleep(1)
        nex.next = head
        #head.next = None
        head = nex
        nex = tmp
        tmp = tmp.next
        #print(head.val, nex.val, tmp.val)
    nex.next = head
    newHead.next = None

    return nex

#a = build_list([1,2,3,4,5])
#print('a-->')
#print_List(a)

#b = reverList(a)
#print('b-->')
#print_List(b)
def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:
    l1Length = 0
    l2Length = 0
    tmphead = l1
    s = []
    while tmphead:
        l1Length += 1
        tmphead = tmphead.next
    tmphead = l2
    while tmphead:
        l2Length += 1
        tmphead = tmphead.next

    if l1Length >= l2Length:
        fast = l1
        slow = l2
    else:
        fast = l2
        slow = l1

    for _ in range(abs(l1Length - l2Length)):
        s.append(fast.val)
        fast = fast.next
    print(fast.val, slow.val)


    while fast and slow:
        s.append(fast.val + slow.val)
        fast = fast.next
        slow = slow.next

    print(s)
    t = s[::-1] + [0]
    print(t)
    for i in range(len(t)):
        #print(t[i])
        while t[i] >= 10:
            #print(1)
            t[i + 1] += t[i] // 10
            t[i] = t[i] % 10

    if t[-1] == 0:
        t= t[:-1]
        newHead = None
    else:
        newHead = ListNode(-1)

    if l1Length >= l2Length:
        if newHead:
            newHead.next = l1
            #tmphead = newHead
            l1 = newHead
        tmphead = l1

        for i in t[::-1]:
            tmphead.val = i
            tmphead = tmphead.next

        while l1:
            print(l1.val)
            l1 = l1.next
        #return l1

    else:
        if newHead:
            newHead.next = l2
            # tmphead = newHead
            l2 = newHead

        tmphead = l2
        for i in t[::-1]:
            tmphead.val = i
            tmphead = tmphead.next
        while l2:
            print(l2.val)
            l2 = l2.next
        #return l2



    #print(l1Length, l2Length)
    #pass

def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:

    first = reverList(l1)
    second = reverList(l2)
    head = first
    tmp = 0
    while first and second:
        print(f'[start]  first {first.val}, second {second.val}, tmp {tmp}')
        if (first.val + second.val + tmp) // 10 < 1:
            first.val = first.val + second.val + tmp
            tmp = 0
        else:
            #print(f'first {first.val}, second {second.val}, tmp {tmp} , next tmp {(first.val + second.val) // 10}')
            res = (first.val + second.val + tmp) % 10
            tmp = (tmp + first.val + second.val) // 10
            first.val = res
            #print(f'[else]  first {first.val}, second {second.val}, tmp {tmp}',tmp + (first.val + second.val) // 10, tmp, (first.val + second.val) // 10)
        #print(f'[end]   first {first.val}, second {second.val}, tmp {tmp}')

        if not first.next:
            lastFirst = first

        if not second.next:
            lastSecond = second
        first = first.next
        second = second.next


    while first:
        print(f'[start]  first {first.val},  tmp {tmp}')
        if (first.val + tmp) // 10 < 1:
            first.val = first.val + tmp

            tmp = 0
        else:
            res = (first.val + tmp) % 10
            tmp = (first.val + tmp) // 10
            first.val = res

        if not first.next:
            lastFirst = first

        first = first.next

    #print_List(head)

    while second:
        print('second', second.val, tmp, lastFirst.val)
        if (second.val + tmp) // 10 < 1:
            second.val = second.val + tmp
            tmp = 0
        else:
            res = (second.val + tmp) % 10
            tmp = (second.val + tmp) // 10
            second.val = res
            print(second.val, res, tmp)
        lastFirst.next = second

        lastFirst = lastFirst.next
        second = second.next
        #print('lastFirst',lastFirst.val)
    print(print_List(head))
    print('tmp',tmp)
    if tmp!=0:
        lastFirst.next = ListNode(tmp)
    print(print_List(head))
    head = reverList(head)
    print('end')
    print(print_List(head))


# l1 = build_list([1,2,3])
# l2 = build_list([2,7,8,9])
# addTwoNumbers(l1,l2)
def insert(self, head: ListNode, insertVal: int):
    first = head.val
    second = head.next.val
    Head = head
    while insertVal < first:
        head = head.next
        first = head.val
        second = head.next.val
        if second.val < first.val and second.val != Head.val:
            #   到了断层部分
            break
    while second < insertVal:
        head = head.next
        second = head.next.val

    #if second.val == Head.val:
    tmp = head.next.next
    head.next = ListNode(insertVal)
    head.next.next = tmp
    return Head

#   28
'''
[1,2,3,4,5,6,null]
[null,null,7,8,9,10,null]
[null,11,12,null]

[1,2,3,4,5,6,null,null,null,7,8,9,10,null,null,11,12]
[1,2,3,7,8,11,12,9,10,4,5,6]
class Node:
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child
'''
def flatten(head):
    path = []
    def dfs(head):
        HEAD = head
        head = head.child
        while head and head.next:
            if head.child:
                head = dfs(head)
            else:
                head = head.next


        head.next = HEAD.next
        HEAD.next = HEAD.child
        return HEAD

    HEAD = head
    while head:
        if head.child:
            head = dfs(head)
        head = head.next

    return HEAD


#    return path
#   27
head = build_list([])


def reverseLink(head):
    cur = head
    # if cur.next:
    #     nxt = cur.next
    # else:
    #     return head
    if cur and cur.next:
        nxt = cur.next
        while nxt:
            print(cur.val)

            tmp = nxt.next
            nxt.next = cur
            cur = nxt
            nxt = tmp
        print('over',1)
        print(cur.val)
        head.next = None

    return cur
def isPalindrome(head: ListNode) -> bool:
    if not head.next:
        return True
    def reverseLink(head):
        cur = head
        if cur.next:
            nxt = cur.next
            while nxt:
                tmp = nxt.next
                nxt.next = cur
                cur = nxt
                nxt = tmp

            head.next = None

        return cur

    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next


    splithead = reverseLink(slow.next)

    slow.next = splithead
    slow = slow.next

    while slow:

        if head.val == slow.val:
            head = head.next
            slow = slow.next
        else:
            return False

    return True

#print(isPalindrome(head))

#print_List(reverseLink(head))
# print('end')
#   17

s = "aa"
t = "aa"
def minWindow(s: str, t: str) -> str:
    d = defaultdict()
    S = F = 0
    length = float('inf')
    for i in t:
        d[i] = 0
    slow = fast = 0

    while slow < len(s):

        #print('START STATES',s[slow], d)
        if 0 in d.values():


            if fast < len(s):
                if s[fast] in d:
                    d[s[fast]] += 1
                fast += 1
                #print('if',s[fast-1], s[fast-1] in d, d, fast, slow)

                #print('1')
            else:
                if s[slow] in d:
                    d[s[slow]] -= 1
                slow += 1

        else:

            if length >= fast - slow:
                S = slow
                F = fast
                length = min(length, fast - slow)
            if s[slow] in d:
                d[s[slow]] -= 1
            slow += 1
        #print('while','slow--fast->',s[slow:fast],'S-F>>>', s[S:F], d, slow, fast)
        #time.sleep(0.5)
    #print(slow,fast,0 in d.values(), S,F)
    #print('end')
    print(S,F)
    return s[S:F]

#   14
'''
# 输入: s1 = "ab" s2 = "eidboaooo"
# 输出: True
'''
s1 = "ab"
s2 = "ab"
s2 = "abecbabo"

# s1 = "adc"
# s2= "dcda"

# s2 = "cbaebabacd"
# s1= "abc"


r = []
def checkInclusion(s1, s2) -> bool:
    if len(s1) > len(s2):
        return False
    d = collections.defaultdict()
    n = len(s1)
    for i in s1:
        d[i] = 0
    for i in s1:
        d[i] += 1
    print(d)
    slow = fast = length = 0
    for _ in range(n):
        if s2[fast] in d :
            if d[s2[fast]] > 0:
                length += 1
            d[s2[fast]] -= 1
        fast += 1
        print(length, s2[slow:fast],s2[slow], s2[fast],d)
    print('---------------------')
    if length == n:
        r.append(slow)
        #print(True)
        #return True
    while fast < len(s2):

        print(d)

        if s2[fast] in d:
            if d[s2[fast]] > 0:
                length += 1
            d[s2[fast]] -= 1

        fast += 1

        print('fast',length, s2[slow:fast], s2[slow],d)
        if s2[slow] in d:
            print(s2[slow], d[s2[slow]])
            if d[s2[slow]] >= 0: #   导致ab--ecboao错误
                length -= 1
            #length -= 1
            d[s2[slow]] += 1

        slow += 1
        print('slow', length, s2[slow:fast], s2[slow], d, fast - slow)
        if length == n and fast - slow == len(s1):
            r.append(slow)
            # print(True)
            # return True
    print('end')
    # return False
    print(r)
#checkInclusion(s1,s2)


#   16
s = "abcabcbb"

s = "pwwkew"
s = "tmmzuxt"
#s = "aabaab!bb"
def lengthOfLongestSubstring(s: str) -> int:
    slow = fast = length = 0
    seen = collections.defaultdict()
    while fast < len(s):
        print(seen, s[slow:fast+1], s[fast], s[fast] in seen)
        if s[fast] not in seen:
            seen[s[fast]] = fast

        else:
            slow = max(slow,seen[s[fast]]+1)
            seen[s[fast]] = fast


        fast += 1

        length = max(length, fast - slow)

        print(length)

#lengthOfLongestSubstring(s)
#   12
nums = [1,7,3,6,5,6]
# nums = [1, 2, 3]
#nums = [2, 1, -1]
#nums = [-1,-1,-1,-1,-1,-1]
#nums = [-1,-1,-1,-1,-1,0]
#nums = [-1,-1,0,0,-1,-1]
nums = [-1,-1,0,1,1,0]
def pivotIndex(nums) -> int:
    total = sum(nums)
    r = 0
    if total - nums[0] == 0:
        return 0
    for i in range(1,len(nums)):
        r += nums[i-1]

        if total - nums[i] == 2 * r:
            return i
        print(nums[i], r, total - nums[i])

    return -1
    # left = 0
    # right = len(nums)-1
    # sum = nums[left] - nums[right]
    # while left < right:
    #     print(sum, left, right)
    #     if sum > 0:
    #
    #         right -= 1
    #         sum -= nums[right]
    #
    #     elif sum < 0:
    #
    #         left += 1
    #         sum += nums[left]
    #     else:
    #         left += 1
    #         right -=1
    #         sum = nums[left] - nums[right]
    #
    #
    # print(sum, left,right)
    # if left > right:
    #     return -1
    # if sum == 0:
    #     return left
    # else:
    #     return -1

#print(pivotIndex(nums))

#   11
nums = [1]
#nums = [0,1,0]
#nums = [1,1,1,1,1,1,1,0,0,0,0,1,1,0,1,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,1,1,1,1,1,0,0,1,0,1,1,0,0,0,1,0,0,0,1,1,1,0,1,1,0,1,0,0,1,1,0,1,0,0,1,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1]
nums = [0,0,1,0,0,0,1,1]
nums = [0,1]
def findMaxLength(nums) -> int:
    sum = [0]*len(nums)
    d = collections.defaultdict()
    for i in range(len(nums)):
        if nums[i] == 0:
            nums[i] = -1
        sum[i] = sum[i-1] + nums[i]
        if sum[i] not in d and sum[i] !=0:
            d[sum[i]] = i
    #print(nums)
    print(sum)
    left = 0
    right = len(sum)-1
    length = 0
    res = sum[left] - sum[right]
    while right > 0:


        if sum[right] in d:
            length = max(length, right - d[sum[right]])
        if sum[right] == 0:
            length = max(length, right+1)
        right -= 1
    # while left < right:
    #     if sum[left] == sum[right] and sum[left] !=0:
    #         length = max(length,right - left)
    #     if sum[right] == 0:
    #         length = max(length, right+1)
    #     if sum[left] ==0:
    #         length = max(length, left+1)
    #
    #     if res < 0:
    #         left += 1
    #         res += sum[left]
    #     else:
    #         right -= 1
    #         res -= sum[right]
        print(f'{left} = {sum[left]}, {right} = {sum[right]} , length = {length} ')
    print(left, right ,length)

#findMaxLength(nums)
nums = [1,2,3]
k = 3

nums = [1,1,1]
k = 2

nums = [1]
k = 1

# nums = [-1,-1,1]
# k = 2
#
# nums = [-1,-2,-3]
# k = -3
#
# nums = [1,2,1,2,1]
# k = 3
#
# nums = [-1,-1,1]
# k = 0
def subarraySum(nums, k) -> int:
    total = nums
    d = collections.defaultdict()
    d[total[0]] = 1

    if nums[0] == k:
        res = 1
    else:
        res = 0


    for i in range(1,len(nums)):
        total[i] = total[i-1] + nums[i]
        if total[i] - k in d:
            res += d[total[i] - k]

        if total[i] == k:
            res += 1
        if total[i] not in d:
            d[total[i]] = 1
        else:
            d[total[i]] += 1


            print(total)
            print(d)
    #print(1 in d)
    # for i in range(len(total)):
    #     print(total[i],total[i] -k, total[i]==k, total[i]-k in d)
    #     if total[i] == k:
    #         res += 1
    #     if total[i] - k in d :
    #         res += d[total[i] - k]

    print(res)

# def subarraySum( nums, k: int):
#     n = len(nums)
#     hashmap = defaultdict()
#     ans = 0
#     preSum = [0 for i in range(n + 1)]
#     hashmap[0] = 1
#     for i in range(1, n + 1):
#         preSum[i] = preSum[i - 1] + nums[i - 1]
#         print(i, hashmap, preSum[i] - k)
#         if preSum[i] - k in hashmap:
#             print(i,hashmap, preSum[i] - k,hashmap[preSum[i] - k])
#             ans += hashmap[preSum[i] - k]
#         if preSum[i] in hashmap:
#             hashmap[preSum[i]] += 1
#         else:
#             hashmap[preSum[i]] = 1
#
#     print(preSum)
#     print(hashmap)
#     print(ans)
#     return ans



#subarraySum(nums,k)

#   09
nums = [10,5,2,6]
k = 100
#k = 0
nums = [1,2,3]
k = 0
# nums =[10,9,10,4,3,8,3,3,6,2,10,10,9,3]
# k = 19
# nums = [1,1,1]
# k = 2

# nums = [10,3,3,7,2,9,7,4,7,2,8,6,5,1,5]
# k = 30


def numSubarrayProductLessThanK(nums, k):

    left = 0
    right = 0
    n = 0

    total = 1
    ret = 0

    for right, num in enumerate(nums):
        total *= num
        while left <= right and total >= k:
            #print('!')
            total //= nums[left]
            left += 1
            #print(left)
        if left <= right:
            print(nums[left:right+1], right - left + 1)
            ret += right - left + 1

    print('--------------------------------------')
    left = 0
    right = 0
    n = 0



    total = 1


    for right in range(len(nums)):
        total *= nums[right]
        while total >= k and left < len(nums)-1:

            total /= nums[left]
            left += 1

        #if (total[right] < k and left == 0) or (total[right] / total[left - 1] < k and left > 0):
        if total < k:
            n += right - left + 1
            print(nums[left:right+1], right -left + 1)
            #continue

    print('------------------------------------------------------')
    print(ret)
    print(nums[left],nums[right], nums[left:right+1])
    print(n)

#numSubarrayProductLessThanK(nums, k)

#   8

def minSubArrayLen(target: int, nums) -> int:
    length = float('inf')
    left = right =0
    total = 0
    for right in range(len(nums)):
        total += nums[right]
        # if total >= target:
        #     length = min(length, right - left + 1)
        #     print(length, right-left+1, nums[left:right+1])
        while total >= target and left <=right:
            length = min(length, right - left + 1)
            total -= nums[left]
            left += 1

    print(length)

target = 7
nums = [2,3,1,2,4,3]

target = 4
nums = [1,4,4]

target = 11
nums = [1,1,1,1,1,1,1,1]
#minSubArrayLen(target, nums)

#   7
def threeSum(nums: list) ->list:
    nums = sorted(nums)
    res =[]
    for start in range(len(nums)-2):

        if start >0 and nums[start] == nums[start-1]:
            continue
        left = start+1
        right = len(nums)-1
        while left < right:
            print(nums[left:right+1], nums[start] + nums[left] + nums[right], start,left,right)
            # if left < right -1 and nums[left] == nums[left+1]:
            #     left +=1
            #     continue
            if nums[start] + nums[left] + nums[right] <0:
                left += 1
            elif nums[start] + nums[left] + nums[right] > 0:
                right -=1
            else:
                #if [nums[start],nums[left],nums[right]] not in res:
                res.append([nums[start],nums[left],nums[right]])
                # print(start,left,right)
                # res.append([nums[start],nums[left],nums[right]])
                #break
                left +=1
                right -=1
                while left < right  and nums[left] == nums[left-1]:
                    left +=1
            #time.sleep(0.2)
    print(res)
    print(nums)
    return 0


    print(ret)

nums = [-1,0,1,2,-1,-4] #   [[-1,-1,2],[-1,0,1]]
nums =[-2,0,1,1,2]
nums = [0,0,0,0,0,0]
# nums = [-2,-1,1,2,3]
#threeSum(nums)

#   1
def divide(a: int, b: int) -> int:
    res = 0
    for i in range(abs(a)+1):
        if i >= 1 and (i-1)*abs(b) <= abs(a) and i*abs(b) > abs(a):
            res = i-1
            print(i * abs(b),i,abs(b))
            break
        elif i >= 1 and i*abs(b) == abs(a) :
            res = i
            break
        elif i ==0 and abs(a) ==0:
            res = 0
            break

    print('res-->',res)
    if a > 0 and b > 0:
        return res
    else:
        return - res

a = -1
b = 1
#print(divide(a,b))

#   3
def countBits(n: int):
    res = [0, 1, 1]
    for i in range(3, n + 1):
        if i & 1 != 1:

            res.append(res[i >> 1])
        else:

            res.append(res[i - 1] + 1)
        print(i, res)
n = 5
#countBits(n)
# 1010
# 1001
#
# 1000

#   44
#def findNthDigit(n: int) -> int:


#   49
def nthUglyNumber(n: int) -> int:
    dp = [0]*n
    dp[0] = 1
    a,b,c = 0,0,0
    for index in range(1,n):
        dp[index] = min(2*dp[a], 3*dp[b], 5*dp[c])
        print(dp)
        if dp[index] == 2*dp[a]:
            a += 1
        if dp[index] == 3*dp[b]:
            b += 1
        if dp[index] == 5*dp[c]:
            c += 1

    return dp[-1]



#print(nthUglyNumber(1))

# 4
def singleNumber(nums) -> int:
    res = [0]*32

    for num in nums:
        #print(num)
        for index in range(32):
            if num & 1 == 1:
                res[index] += 1
            num >>= 1
            if num == 0:
                break

    for i in range(len(res)):
        res[i] = str(res[i]%3)

    if int(res[-1]) == 1:
        res = int(''.join(res[::-1]), 2)
        res = -((res ^ 0xffffffff)+1)
    else:
        res = int(''.join(res[::-1]), 2)

    return res

#   5
def maxProduct(words) -> int:

    record = []
    MAX = 0
    for word in words:
        res = ['0']*27
        for char in word:
            res[ord(char) - ord('a')] = str(1)
        print(res)

        record.append(int(''.join(res[::-1]),2))

    for i in range(len(words)):
        for j in range(i+1, len(words)):
            if record[i] & record[j] ==0:
                MAX = max(MAX, len(words[i]) * len(words[j]))

    return MAX
    print(record)
    print(MAX)

#  || 17
s = "ADOBECODEBANC"
t = "ABC"
#
# s = "abc"
# t = "b"
#
# s = "aaa"
# t = "a"
# s = 'bba'
# t = 'ab'

s = 'a'
t = 'b'
def minWindow(s: str, t: str) -> str:
    if len(t) > len(s):
        return ""
    d = defaultdict()
    slow =  0

    for char in t:
        if char not in d:
            d[char] = 1
        else:
            d[char] += 1
    total = len(t)
    length = float('inf')
    S,F = -1,-1
    for fast in range(len(s)):
        print(slow, fast, total, d)

        if s[fast] in d:
            if d[s[fast]] > 0:
                total -= 1
            d[s[fast]] -= 1

        while total == 0:
            print('total',slow, fast, total)
            if total == 0 and fast - slow <= length:
                S, F = slow, fast
                length = fast - slow
                print(s[S:F + 1])


            if s[slow] in d:
                if d[s[slow]] >= 0:
                    total += 1
                d[s[slow]] += 1


            slow += 1


    if S >= 0:
        return s[S:F+1]
    else:
        return ''

#print('res->',minWindow(s,t))

# #面试题 17.18. 最短超串
#
#
# def shortestSeq(big, small):
#
# big = [7, 5, 9, 0, 2, 1, 3, 5, 7, 9, 1, 1, 5, 8, 8, 9, 7]
# small = [1, 5, 9]


#   32
def isAnagram(s: str, t: str) -> bool:
    d = collections.defaultdict()
    counter =  0
    for i in s:
        if i not in d:
            d[i] = 1
        else:
            d[i] += 1

    for i in t:
        if i in d:
            d[i] -= 1
        else:
            return False

        if d[i] < 0:
            return False

    # for i,j in zip(s,t):
    #     if d[i] ==0:
    #         if i ==j :
    #             counter += 1
    #             continue
    #         else:
    #             counter = 0
    #     else:
    #
    #         return False
    #
    # if counter == len(s):
    #
    #     return False
    # else:

    #   33
    for i in d:
        if d[i] ==0:
            continue
        else:
            return False
    return True

    #print(d)

s = "anagram"
t = "nagaram"
# 输出: true
s = "rat"
t = "car"

s = "a"
t = "a"


s = "ab"
t = "a"
#res = isAnagram(s,t)

#print(res)

def groupAnagrams(strs) :
    hash = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97, 101,
    ]
    chars = collections.defaultdict(list)
    for s in strs:
        total = 1
        for i in s:
            total *= hash[ord(i) - ord('a')]
        chars[total].append(s)

    return list(chars.values())


strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
# 输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
# strs = ["", ""]
# strs = ['s','s']
strs = ["bdddddddddd","bbbbbbbbbbc"]
#groupAnagrams(strs)




#print('b' - 'a')


#print(pow(10,100))

# import pandas as pd
# import os
# MAX_INT = 1000
# threshold = 0.6
# pd.set_option('expand_frame_repr', False)
# pd.set_option('display.max_rows', MAX_INT)
# pd.set_option('display.max_columns', MAX_INT)
# data = pd.read_csv(r'C:/Users/w8856/Desktop/data.csv',encoding = 'utf-8')
#
# data = data.dropna(axis=0)
# myboy = data.loc[data['姓名'] == '郑自强']
# #myboy = data.loc[data['姓名'] == '安天一']
# # print(data.shape)
# print(myboy)
#
#
#
# No4_class = data.loc[data['班级'] == myboy.iloc[0]['班级']]
# No4_class = No4_class.sort_values('学位课加权平均分',ascending=False).reset_index()
# #
# NUM = int(No4_class.shape[0] * threshold)
# #
# #print(No4_class[:NUM])
# # print(No4_class.shape[0], NUM)
#
#
# ProMaster_class = data.loc[data['学生类别'] == myboy.iloc[0]['学生类别']]
#
# ProMaster_class = ProMaster_class.sort_values('学位课加权平均分',ascending=False).reset_index()
# NUM = int(ProMaster_class.shape[0] * threshold)
#
# #print(ProMaster_class[:NUM])
#
#
# AcaMaster_class = data.loc[data['学生类别'] == '学术型硕士']
# #print(AcaMaster_class.shape)
# #data = data.reset_index()
#
# print(f"{myboy.iloc[0]['姓名']}\t在 {No4_class.iloc[0]['班级']} 排名 "
#       f"{No4_class[No4_class['姓名'] == myboy.iloc[0]['姓名']].index.values[0] + 1} / {No4_class.shape[0]} , 截至 {int(No4_class.shape[0] * threshold)} \n"
#       f"        在 {ProMaster_class.iloc[0]['学生类别']} 排名 "
#       f"{ProMaster_class[ProMaster_class['姓名'] == myboy.iloc[0]['姓名']].index.values[0] + 1}/{ProMaster_class.shape[0]}, 截至"
#       f"{int(ProMaster_class.shape[0] * threshold)} \n"
#       f"        在 {data.iloc[0]['年级']} 排名 {data[data['姓名'] == myboy.iloc[0]['姓名']].index.values[0] + 1} / {data.shape[0]} 截至"
#       f"{int(data.shape[0] * threshold)}"
#       )

#   34
def isAlienSorted(words, order):
    dic = defaultdict()

    temp = 0
    for i,c in enumerate(order):
        dic[c] = i

    if len(words) < 2:
        return True

    for start_word in range(len(words)-1):
        tail_word = start_word + 1

        for i,j in zip(words[start_word], words[tail_word]):
            if dic[i] < dic[j]:
                temp = 0
                break
            elif dic[i] > dic[j]:
                temp = 0
                return False
            else:
                temp =1
                continue

        if len(words[start_word]) > len(words[tail_word]) and temp == 1:
            print(1)
            return False

    return True


words = ["hello", "leetcode"]
order = "hlabcdefgijkmnopqrstuvwxyz"

words = ["word","world","row"]
order = "worldabcefghijkmnpqstuvxyz"

words = ["apple","app"]
order = "abcdefghijklmnopqrstuvwxyz"

# words = ['a']
# order = "a"
#
words = ["kuvp","q"]
order = "ngxlkthsjuoqcpavbfdermiywz"
#print(isAlienSorted(words, order))

#   35
def findMinDifference(timePoints) -> int:

    timePoints = sorted(timePoints)
    MIN = 10000
    for i in range(len(timePoints)):
        #print()
        timePoints[i] = int(timePoints[i].split(":")[0]) * 60 + int(timePoints[i].split(":")[-1])

    for i in range(1, len(timePoints)):
        MIN = min(24*60 - timePoints[i] + timePoints[i-1], timePoints[i] - timePoints[i-1], MIN)

    if len(timePoints) > 2:
        MIN = min(24*60 - timePoints[-1] + timePoints[0], MIN)
    print(timePoints, MIN)
    return MIN



timePoints = ["00:00","23:59","00:00"]
timePoints = ["23:59","00:00"]

timePoints = ["00:00","04:00","22:00"]
#print(findMinDifference(timePoints))

#   36
def evalRPN(tokens) -> int:
    s = []

    for i in tokens:

        if i == '+':
            s.append(s.pop() + s.pop())
        elif i == '-':
            first = s.pop()
            second = s.pop()
            s.append(second - first)
        elif i == '*':
            s.append(s.pop() * s.pop())
        elif i == '/':
            first = s.pop()
            second = s.pop()
            s.append(int(second / first))
        elif -200 <= int(i) <= 200:
            s.append(int(i))

    return s[-1]


tokens = ["2","1","+","3","*"]
tokens = ["4","13","5","/","+"]
tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
#print(evalRPN(tokens))


#   37
def asteroidCollision(asteroids):
    s = []
    for i in range(len(asteroids)):



        if len(s)!=0 and s[-1]* asteroids[i] <0:
            print(s[-1], asteroids[i], s[-1] < 0 and asteroids[i] > 0)
            #if (s[-1] + asteroids[i] )*s[-1] <= 0:
            if s[-1] < 0 and asteroids[i] > 0:
                s.append(asteroids[i])
                continue
            else:
                cur = asteroids[i]
                #j = len(s) -1
                print('s----------->',s, cur)
                while s:
                    if (s[-1] > 0 and cur < 0 ):
                        if abs(s[-1]) < abs(cur):
                            s.pop()
                        elif abs(s[-1]) == abs(cur):
                            s.pop()
                            cur = 0
                            break
                        else:
                            cur = 0
                            break
                    else:
                        break
                if cur !=0:
                    s.append(cur)
                print('!s------>',s)

        elif len(s) ==0 or s[-1] * asteroids[i] >0:
            #print(asteroids[i])
            s.append(asteroids[i])
        #print(s)

    return s

asteroids = [-2,-1,1,2]
#asteroids = [10,2,-5]
asteroids = [2,-2]
asteroids = [5,10,-5]
asteroids = [10,2,-5]
asteroids = [-2,-1,1,2]
asteroids = [-2,-2,1,-1]
asteroids = [1,-2,-2,-2]
#asteroids = [10,2,-5]
asteroids = [-2,-2,1,-2]
#print(asteroidCollision(asteroids))

#   38
def dailyTemperatures(temperatures):
    s = [0] * len(temperatures)
    res = [0] * len(temperatures)
    top = -1
    for i in range(len(temperatures)):
        t = temperatures[i]

        while top >= 0 and temperatures[s[top]] < t:
            res[s[top]] = i - s[top]
            top -= 1


        top += 1
        s[top] = i

    return res
temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
# temperatures = [30,40,50,60]
# temperatures = [30,60,90]
# temperatures = [30]

#print(dailyTemperatures(temperatures))


#   41
class MovingAverage:

    def __init__(self, size: int):
        """
        Initialize your data structure here.
        """
        self.s = [0]* size
        self.size = size
        self.total = 0
        self.num = 0
    def next(self, val: int) -> float:
        self.total += val - self.s[self.num % self.size]
        self.s[self.num % self.size] = val
        self.num += 1

        #self.total = sum(self.s)
        return self.total / min(self.num, self.size)
        # if self.num > self.size:
        #     self.total += val - self.s[0]
        #     for i in range(self.size):
        #         self.s[i] = self.s[i+1]
        #     return self.total / self.size
        # else:
        #     self.total += val
        #     return self.total / self.num





# obj = MovingAverage(3)
# l = [1,10,3,5,1,2]
# #print(l[2:3])
# for i in l:
#     print(obj.next(i))
# param_1 = obj.next(1)
# print(param_1)
# param_1 = obj.next(10)
# param_1 = obj.next(3)
# param_1 = obj.next(5)

#   43
# obj = CBTInserter(root)
# param_1 = obj.insert(v)
# param_2 = obj.get_root()
class CBTInserter:

    def __init__(self, root: TreeNode):

        self.cur = root
        self.seq = collections.defaultdict({1:root})
    def insert(self, v: int) -> int:
        #self.seq.append(TreeNode(v))
        if len(self.seq) % 2==0:
            self.seq[len(self.seq) // 2].left = TreeNode(v)
        else:
            self.seq[len(self.seq) // 2].right = TreeNode(v)

        self.seq[len(self.seq) + 1] = TreeNode(v)

    def get_root(self) -> TreeNode:
        return self.seq[1]


#   44
class Solution:
    def largestValues(self, root: TreeNode):
        res = []
        cur = [root]
        MAX = -float('inf')
        while cur:
            next_layer = []
            for i in cur:
                MAX = max(MAX, i.val)
                if i.left:
                    next_layer.append(i.left)
                if i.right:
                    next_layer.append(i.right)
            cur = next_layer
            res.append(MAX)
        return res


#   39
# 输入：heights = [2,1,5,6,2,3]
# 输出：10
def largestRectangleArea(heights):
    s = []
    h = 0
    for i in range(len(heights)):
        # if len(s) >0 :
        #     print(i,s[-1],  heights[i])
        print(i, heights[i])
        if len(s) == 0 or heights[s[-1]] <= heights[i]:
            s.append(i)
        #elif heights[s[-1]] > heights[i]:
        else:
            print(len(s) == 0 , heights[s[-1]] <= heights[i])
            while len(s) > 0 and heights[s[-1]] > heights[i]:
                print('start-->',[heights[i] for i in  s], 'h---->',h)
                if len(s) >= 2:
                    print(heights[s[-1]] * (i - s[-1-1] -1),heights[s[-1]] , (i - s[-1-1] -1))
                    h = max(h, heights[s[-1]] * (i - s[-1-1] -1))
                else:
                    print(heights[s[-1]] * i, heights[s[-1]] , i)
                    h = max(h, heights[s[-1]] * i)
                s = s[:-1]
                print([heights[i] for i in s] , 'h--->',h)
            s += [i]
            print('end-->', [heights[i] for i in s] , 'h--->',h)
        # else:
        #     print('!!!')
        #     print(len(s) == 0, heights[s[-1]] <= heights[i])

    #s = s[1:]
    print('s--->',s, [heights[i] for i in s], h)
    for index,i in enumerate(s):
        print(heights[i], s[-1], i)
        if index!=0:
            print(f'index != 0 h : {h}, height[i] : {heights[i]}, s[-1] : {s[-1]}, s[index -1 ]: {s[index-1]}, windowsize: {s[-1] - s[index -1]}')
            h = max(h, heights[i] * (s[-1] - s[index -1]))
        else:
            print(f'index == 0 h : {h}, height[i] : {heights[i]}, len(heights) : {len(heights)}')
            h = max(h, heights[i] * len(heights))

    return h


heights = [2,1,5,6,2,3]
# heights = [2,4]

#heights = [1,2,2]
heights = [1,2,3,4,5]
#heights = [2,1,2]
#heights = [4,2,0,3,2,5]
#print('res-->',largestRectangleArea(heights))

#   40

def rectangle(matrix, startX, startY, endX, endY):


    res = []
    for i in range(startX, endX+1):
        temp = []
        for j in range(startY, endY+1):
            temp.append(matrix[i][j])

        res.append(temp)
    print(startX, startY, endX, endY, '\n',np.array(res))
    for i in range(startX, endX+1):
        for j in range(startY, endY+1):
            if matrix[i][j] == '0':
                print(f'res = {0}')
                return 0
    print(f'res = {(endX+1 - startX)*(endY+1 - startY)}')
    return (endX+1 - startX)*(endY+1 - startY)


def maximalRectangle(matrix):

    res = 0
    #print(len(matrix), len(matrix[0]))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            h = 0
            while i + h < len(matrix):
                l = 0
                while j + l < len(matrix[0]):

                    #print(i,j, i+h, j+l)
                    res = max(res, rectangle(matrix,i,j,i+h,j+l))
                    #print(res)
                    if res == 0:
                        break
                    l += 1
                h += 1
    return res

matrix = ["10100","10111","11111","10010"]
#matrix = []
#matrix = ["0"]
#matrix = ["1"]
#matrix = ["01"]
#matrix = ["001","111"]
#matrix = ["1111111111111101001111111100111011111111","1111011011111111101101111101111111111111","0111101011111101101101101111111111111111","0101101011111111111101111111010110111111","1111111111110111110110010111111111111111","1111111110110101111111111101011111101111","0110110101110011111111111111110111110101","0111111111111100111111100110011011010101","1111011110111111111011011011110101101011","1111111110111111111101101101110111101111","1110110011111111111100111111111111111111","1011111110111101111001110111111111111111","0110111111111111101111110111011111011111","1111111111111111011111111111110111111011","1111100111111110101100111111111111101111","1111101111111110111111011111111111011111","1111101111111111111111011001111110011111","1111110111111111011111111111110111110111","1011111111111111010111110010111110111111","1111110011111111111110111111111111111011","1111111111111111110111011111011111011011","1100011011111111111111011111011111111111","1111101011111111111101100101110011111111","1110010111111111111011011110111101111101","1111111111111101101111111111101111111111","1111111011111101111011111111111110111111","1110011111111110111011111111110111110111","1111111111111100111111010111111111110111","1111111111111111111111000111111111011101","1111110111111111111111111101100111011011","1111011011111101101101111110111111101111","1111111111011111111111111111111111111111","1111111111111111111111111111111111111111","1100011111111110111111111111101111111011","1111101111111101111010111111111111111111","0111111111110011111111110101011011111111","1011011111101111111111101111111111110011","1010111111111111111111111111111110011111","0111101111011111111111111111110111111111","0111111011111111011101111011101111111111","0111111111110101111011111101011001111011","1111111111111011110111111101111111101110","1111101111111100111111111110111111001111","1101101111110101111101111111100111010100","0110111111100111110010111110111011011101"]
matrix = ["0000001","0000111","1111111","0001111"]

matrix = ["01101","11010","01110","11110","11111","00000"]
matrix = ["0110111111111111110","1011111111111111111","1101111111110111111","1111111111111011111","1111111111111101111","1110111011111111101","1011111111111101111","1111111111111110110","0011111111111110111","1101111111011111111","1111111110111111111","0110111011111111111","1111011111111101111","1111111111111111111","1111111111111111111","1111111111111111101","1111111101101101111","1111110111111110111"]

def largestRectangleArea(matrix):
    # heights = matrix
    # s = []
    # h = 0
    # for i in range(len(heights)):
    #     # if len(s) >0 :
    #     #     print(i,s[-1],  heights[i])
    #     print(i, heights[i])
    #     if len(s) == 0 or heights[s[-1]] <= heights[i]:
    #         s.append(i)
    #     # elif heights[s[-1]] > heights[i]:
    #     else:
    #         print(len(s) == 0, heights[s[-1]] <= heights[i])
    #         while len(s) > 0 and heights[s[-1]] > heights[i]:
    #             print('start-->', [heights[i] for i in s], 'h---->', h)
    #             if len(s) >= 2:
    #                 if heights[s[-1]] * (i - s[-1 - 1] - 1) >= h:
    #                     print('if!!!!!!!!!!!!',matrix, heights[s[-1]] * (i - s[-1 - 1] - 1))
    #                 #h = max(h, heights[s[-1]] * (i - s[-1 - 1] - 1))
    #             else:
    #                 print(heights[s[-1]] * i, heights[s[-1]], i)
    #                 if heights[s[-1]] * i > h:
    #                     print('else!!!!!!!!!!!!!!',matrix, heights[s[-1]] * i)
    #                 h = max(h, heights[s[-1]] * i)
    #             s = s[:-1]
    #             print([heights[i] for i in s], 'h--->', h)
    #         s += [i]
    #         print('end-->', [heights[i] for i in s], 'h--->', h)



    s = []
    area = 0
    for i in range(len(matrix)):
        #print([matrix[i] for i in s], i, matrix[i])
        if len(s) == 0 or matrix[s[-1]] < matrix[i]:
            s.append(i)
            #print('if',[matrix[i] for i in s], i, matrix[i])
        elif matrix[s[-1]] >= matrix[i]:
            #print(s[-1], matrix[i])

            while len(s)!=0 and matrix[s[-1]] >= matrix[i] :
                #print('else', [matrix[i] for i in s], i, matrix[i])
                if len(s) >=2:
                    #area = max(s[-1]* (i-s[-1-1]), area)
                    if matrix[s[-1]]* (i-s[-1-1]-1) >= area:
                        area = matrix[s[-1]]* (i-s[-1-1]-1)
                        #print([matrix[i] for i in s], matrix[i])
                        #print(f'larger height : {matrix[s[-1]]} , start : {i}, end : {s[-1]}, area : {area}')
                    else:
                        pass
                        #print(f'height : {matrix[s[-1]]} , start : {i}, end : {s[-1]}, area : {area}')
                else:
                    #print(i, s, i+1-s[-1])
                    print(i, matrix[i])
                    area = max(matrix[s[-1]]* (i), area)
                    #print('else area : ', area)
                s.pop(-1)
            s.append(i)
            #print([matrix[i] for i in s], area)
    print('while', [matrix[i] for i in s], area)
    #area = h

    print(len(s))
    while s:
        if len(s) >= 2:
            #area = max(matrix[s[-1]] * (len(matrix) - s[-1-1] -1), area)
            if matrix[s[-1]] * (len(matrix) - s[-1-1] -1) > area:
                print('whileif!!!!!!!!!!!!!!', matrix, matrix[s[-1]] * (len(matrix) - s[-1-1] -1))
            area = max(matrix[s[-1]] * (len(matrix) - s[-1-1] -1), area)
            #print(matrix[s[-1]], s[-1], s[-1-1], len(matrix) -1)
        else:
            if matrix[s[-1]] * (len(matrix)) > area:
                print('whileelse!!!!!!!!!!!!!',matrix, matrix[s[-1]] * (len(matrix)))
            area = max(matrix[s[-1]] * (len(matrix)), area)

        s.pop(-1)
        #print([matrix[i] for i in s], area)

    return area


#   II 40

# m = [0]*len(matrix[0])
# area = 0
# for i in range(len(matrix)):
#     for j in range(len(matrix[0])):
#         if matrix[i][j] == '1':
#             m[j] += 1
#         else:
#             m[j] = 0
#
#     #print(m)
#     area = max(area, largestRectangleArea(m))
#     print(area, largestRectangleArea(m))
#     if area == 51:
#         print(m)
        #raise Exception

# 17.21

def trap(height) -> int:
    s = []

    c = 0
    for i in range(len(height)):
        print('start',[height[i] for i in s], height[:i+1],c)
        if len(s) == 0 or height[s[-1]] > height[i]:
            s.append(i)
        else:
            # print('else----->', [height[i] for i in s], height[:i + 1], c)
            # if height[s[-1]] == height[i]:
            #     print('if =================', [height[i] for i in s], height[:i + 1], c)
            #     s.pop(-1)
            #     s.append(i)
            # else:
            while len(s)> 0 and height[s[-1]] <= height[i]:
                if len(s)>=2:
                    c += (min(height[s[-1-1]], height[i]) - height[s[-1]])* (i - s[-1])
                    print('if', c)
                else:
                    c += (height[i] - height[s[-1]]) * (i - s[-1] - 1)

                    print('else',c)
                s.pop(-1)

            s.append(i)
        print('end',[height[i] for i in s],height[:i+1], c)
    #print(s)
    return c

# def trap(height):
#     s = []
#
#     for i in range(len(height)):

def trap(height) -> int:
    s = []

    for i in height:
        if i not in s and i != 0:
            s.append(i)
    s.sort()

    left = 0
    right = len(height) - 1

    c = sum(height)
    v = 0
    for h in range(len(s)):
        while height[left] < s[h]:
            left += 1
        while height[right] < s[h]:
            right -= 1

        if v == 0:
            v += (right - left + 1) * s[h]
        else:
            v += (right - left + 1) * s[h] - (right - left + 1) * s[h - 1]
        print(f"v : {v}, left : {left}, right : {right}")
    return v - c

def trap(height) -> int:
    if len(height) == 0:
        return
    res = 0

    dp =[[0]*len(height), [0]*len(height)]
    dp[0][0] = 0
    dp[1][-1] = len(height)-1
    left, right = 0, len(height)-1

    for i in range(1,len(height)):
        if height[i] >= height[dp[0][i-1]]:
            dp[0][i] = i
        else:
            dp[0][i] = dp[0][i-1]

    for i in range(len(height)-1, 0,-1):
        #print(dp[1], height[i-1], height[dp[1][i]], i)
        if height[i-1] >= height[dp[1][i]]:
            dp[1][i-1] = i-1
        else:
            dp[1][i-1] = dp[1][i]

    for i in range(len(height)):
        if dp[0][i] < i and dp[1][i] > i:
            res += min(height[dp[0][i]], height[dp[1][i]]) - height[i]


    return res



heights = [0,1,0,2,1,0,1,3]
#heights = [0,1,0,2,1,0,1,3,2,1,2,1]
heights = [0,2,0,1,2]
heights = [0,1,0,2,1,0,1,3,2,1,2,1]
heights = [2,0,2]
heights = []
#print(trap(heights))

# 15
def threeSum(nums):

    res = []

    nums = sorted(nums)

    print(nums)

    #second, last = 1, len(nums)-1
    for first in range(len(nums)-2):

        if first > 0 and nums[first] == nums[first-1]:
            continue
        if nums[first] > 0:
            break
        second = first + 1
        last = len(nums)-1
        while second < last:
            if nums[second] + nums[last] > -nums[first]:

                last -= 1
            elif nums[second] + nums[last] < -nums[first]:
                second += 1
            else:
                while second < last -1 and nums[second] == nums[second+1]:

                        second += 1

                while second < last -1 and  nums[last] == nums[last-1]:
                        last -= 1

                #print(first, second, last)
                res.append([nums[first], nums[second], nums[last]])
                second += 1
    return res

nums = [-1,0,1,2,-1,-4,-3,-2,-2]
nums = [-1,0,0,1,2,-1,-4]
nums = [0,0,0,0]
nums = [-1,0]
#nums = [-1,0,1,2,-1,-4,-2,-3,3,0,4] # [[-4,0,4],[-4,1,3],[-3,-1,4],[-3,0,3],[-3,1,2],[-2,-1,3],[-2,0,2],[-1,-1,2],
# [-1,0,1]]
# [-4,1,3]
#print(threeSum(nums))

#   18
def fourSum(nums, target):
    res = []
    nums = sorted(nums)

    print(nums)
    for first in range(len(nums)-3):
        if first > 0 and nums[first] == nums[first-1]:
            continue

        if nums[first]+ nums[first+1]+nums[first+2]+nums[first]+3>target:
            break
        for second in range(first+1, len(nums)-2):
            if second > first+1 and nums[second] == nums[second-1]:
                continue
            third = second + 1
            fourth = len(nums)-1

            while third < fourth:
                print(first, second, third, fourth)
                print([nums[i] for i in [first, second, third, fourth]])
                if nums[first] + nums[second] + nums[third] + nums[fourth] > target:
                    fourth -= 1
                elif nums[first] + nums[second] + nums[third] + nums[fourth] < target:
                    third += 1
                else:
                    #print(first, second, third, fourth)
                    while third < fourth - 1 and nums[third] == nums[third + 1]:
                        third += 1
                    while third < fourth - 1 and nums[fourth] == nums[fourth - 1]:
                        fourth -= 1
                    res.append([nums[first],nums[second],nums[third],nums[fourth]])

                    third += 1
    return res

nums = [1,0,-1,0,-2,2]
target = 0
nums = [2,2,2,2,2]
target = 8
nums = [-2,-1,-1,1,1,2,2]
target = 0
nums = [1,-2,-5,-4,-3,3,3,5]
target = -11
#print(fourSum(nums,target))

#   II56
def findTarget(root, k):
    # small, large = root, root
    #
    # while small and large:
    #     if small.val + large.val > k:
    #         if small.left:
    #             small = small.left
    #         else:
    #             return False
    #     elif large.val + large.val < k:
    #         if large.right:
    #             large = large.right
    #         else:
    #             return False
    #     else:
    #         return True
    d = {}
    if root:
        if k - root.val in d:
            return True
        else:
            d[root.val] = 1
    else:
        return False
    return findTarget(root.left, k) or findTarget((root.right, k))

# II23
def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
    d = {}
    while headA and headB:
        if headA in d:
            return headA
        else:
            d[headA] = 1
        if headB in d:
            return headB
        else:
            d[headB] = 1

        headA, headB = headA.next, headB.next

    while headA:
        if headA in d:
            return headA
        else:
            headA = headA.next

    while headB:
        if headB in d:
            return headB
        else:
            headB = headB.next
    return

#  57
def twoSum(nums, target):
    small, large = 0, len(nums)-1

    while small < large:
        if nums[small] + nums[large] == target:
            return [nums[small], nums[large]]
        elif nums[small] + nums[large] > target:
            while small < large:
                print('elif',nums[small:large + 1])
                if  nums[small] + nums[(small + large + 1)//2] > target:
                    large = (small + large + 1)//2
                elif nums[small] + nums[(small + large + 1)//2] == target:
                    return [nums[small], nums[(small + large + 1)//2]]
                else:
                    large -= 1
                    break

        else:
            while small < large:
                #print('else',nums[small:large + 1])
                if nums[(small + large + 1) // 2] + nums[large] < target:
                    small = (small + large + 1) // 2
                elif nums[(small + large + 1) // 2] + nums[large] == target:
                    return [nums[(small + large + 1) // 2], nums[large]]
                else:
                    small += 1
                    break
        #print(nums[small:large+1])
    return False

#print(twoSum(nums = [10,26,30,31,47], target = 40))
nums =[14,15,16,22,53,60]
target =		76

nums =[2,7,11,15]
target=			9
#print(twoSum(nums, target))

# II19
def validPalindrome(s: str):

    flag = 0

    def f(flag, s):
        left, right = 0, len(s) - 1
        while left < right:
            if s[left] == s[right]:
                left +=1
                right -=1
            else:
                if flag == 0:
                    flag = 1
                    return f(flag, s[left+1:right+1]) or f(flag, s[left:right])
                else:
                    return False
        return True

    return f(flag, s)
s = "abc"
s = "abca"
s = 'ab'
#s = "eeccccbebaeeabebccceea"
#s = "aguokepatgbnvfqmgmlcupuufxoohdfpgjdmysgvhmvffcnqxjjxqncffvmhvgsymdjgpfdhooxfuupuculmgmqfvnbgtapekouga"
#print(validPalindrome(s))

# II18
def isPalindrome(s):

    left, right = 0, len(s)-1
    while left < right:
        if s[left].isalnum() and s[right].isalnum():
            if s[left].lower() == s[right].lower():
                left += 1
                right -=1
            else:
                return False
        elif s[left].isalnum():
            right -=1
        elif s[right].isalnum():
            left += 1
        else:
            left += 1
            right -= 1

    return True

s = "A man, a plan, a canal: Panama"
# s = 'a'
# s = "race a car"
s = " "
s = "0P"
#s = "ab_a"

#print(isPalindrome(s))

# II6
def twoSum(numbers, target):
    first, second = 0, len(numbers)-1
    while first < second:
        if numbers[first] + numbers[second] == target:
            return [first, second]
        elif numbers[first] + numbers[second] > target:
            second -= 1
        else:
            first += 1

    return

numbers = [-1,0]
target = -1
#print(twoSum(numbers, target))

# OFFER52
def getIntersectionNode(headA, headB):

    first, second = headA, headB
    while first != second:
        if first == second:
            return first

        first = first.next if first else headB
        second = second.next if second else headA

    return

#   II22
def detectCycle(head):
    slow, fast = head, head
    while slow != fast:
        if fast.next and fast.next.next:
            slow, fast = slow.next, fast.next.next
        else:
            return

    fast = head
    while slow != fast:
        slow, fast = slow.next, fast.next


    return fast

#   977
# 输入：nums = [-4,-1,0,3,10]
# 输出：[0,1,9,16,100]
def sortedSquares(nums):
    res = []
    left, right = -1,0
    while right < len(nums)-1 and abs(nums[right+1]) <= abs(nums[right]) :
        right+=1
        left +=1


    while len(res) < len(nums):
        print('nums:',left, right,  nums[left:right+1])

        if left < 0:
            res.append(nums[right] * nums[right])
            right += 1
        elif right == len(nums):
            res.append(nums[left] * nums[left])
            left -= 1
        elif abs(nums[left])<= abs(nums[right]):
            res.append(nums[left]*nums[left])
            left -= 1
        else:
            res.append(nums[right] * nums[right])
            right += 1


    return res

nums = [-4,-1,0,3,10]
nums = [-7,-3,2,3,11]
nums = [1,2]
nums = [-5,-3,-2,-1]
nums = [-4,-1,0,3,10]
#print(sortedSquares(nums))

# 344
# 输入：s = ["h","e","l","l","o"]
# 输出：["o","l","l","e","h"]
def reverseString(s):

    left,right = 0,len(s)-1
    while left < right:
        s[left] ^= s[right]
        s[right] ^= s[left]
        s[left] ^= s[right]
        left += 1
        right -= 1
    return s

#   2108
def firstPalindrome(words):


    for seq in words:
        left, right, flag = 0, len(seq)-1, 0
        while left < right:
            if seq[left] != seq[right]:
                flag = -1
                break
            else:
                left += 1
                right -= 1
        if flag == 0:
            return seq

    return ""

words = ["abc","car","ada","racecar","cool"]
words = ["notapalindrome","racecar"]
words = ["def","ghi"]
#print(firstPalindrome(words))

# 输入：s = "abpcplea", dictionary = ["ale","apple","monkey","plea"]
# 输出："apple"

def findLongestWord(s, dictionary):

    dictionary = sorted(dictionary, reverse=True)
    print(dictionary)
    length = 0
    res = []
    for word in dictionary:
        left, right = 0, 0
        while left < len(s) and right < len(word):
            if s[left]==word[right]:
                left += 1
                right += 1
            else:
                left += 1

        if right == len(word):
            if len(word) >= length:
                res = word
                length = len(word)

    return res

s = "abpcplea"
dictionary = ["ale","apple","monkey","plea"]
# s = "abpcplea"
# dictionary = ["a","b","c"]
#print(findLongestWord(s, dictionary))

# 283
# 输入: nums = [0,1,0,3,12]
# 输出: [1,3,12,0,0]
def moveZeroes(nums):

    # slow 0 fast !0
    slow, fast = 0,0
    while fast < len(nums) and slow < len(nums):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1
        fast += 1
    return nums

nums = [0,1,0,3,12]
#nums = [0]
#nums = [1]
#nums = [1,0,0]
#print(moveZeroes(nums))


# 611
# 输入: nums = [2,2,3,4]
# 输出: 3
# 解释:有效的组合是:
# 2,3,4 (使用第一个 2)
# 2,3,4 (使用第二个 2)
# 2,2,3

'''
回溯法，超时，需要遍历所有无效数据，没办法剪枝
'''
def triangleNumber(nums):

    nums = sorted(nums)
    print(nums)
    res = []
    path = []
    flag = []
    def dfs(path, startIndex):
        print('res->',res,'path->', path)
        if len(path) + len(nums[startIndex:]) < 3:
            return
        elif len(path) == 3:
            if path[0] + path[1] > path[-1] and path[0] + path[-1] > path[1] and path[-1] + path[1] > \
                path[0]:
                res.append(path[:])
                return
            else:
                flag.append(-1)
                return

        for index in range(startIndex, len(nums)):
            # if index > startIndex and nums[index] == nums[index-1]:
            #     continue
            if nums[index] == 0:
                continue
            path.append(nums[index])
            dfs(path, index+1)
            if len(flag)== 0 or flag[-1] != -1:
                path.pop(-1)
            else:
                path.pop(-1)
                flag.append(0)
                return

    dfs(path, 0)
    print(res)
    return res

def triangleNumber(nums):
    nums = sorted(nums)

    res = []
    n = 0

    for first in range(len(nums)-2):

        if nums[first] == 0:
            continue

        for second in range(first+1, len(nums)-1):
            if nums[second] == 0:
                continue
            l,r = second + 1, len(nums)-1
            k = second
            while l <= r:
                #m = (l + r) // 2
                if nums[first] + nums[second] > nums[(l+r)//2]:
                    k = (l + r) // 2
                    l = (l + r) // 2 + 1
                else:
                    r = (l + r) // 2 - 1

                #print(l,r,(l + r) // 2)
            n += k - second
            print(n)
    return n

def triangleNumber(nums):
    nums.sort()
    n = 0
    for i in range(len(nums)-2):
        left, right = i+1, i+2
        k = left
        while left <= len(nums)-2:
            #print('start ',i, left, right)

            if right < len(nums) and nums[left] + nums[i] > nums[right]:
                right += 1
                k += 1
            else:
                #print('n++\nleft+1')
                if right < len(nums):
                    n += k - left
                    left += 1
                elif nums[left] + nums[i] > nums[k]:
                    n += k - left
                    left += 1
                    print(k, len(nums), right)
            #print('end ', i, left, right, n)
            print(n)
    return n

#nums = [1]
nums =[15,44,16,43,47,47,45,27,46,2,28,12,49,22,36,12,6,48,28,19,18,34,46,38,42,3,21,3,54,35,21,54,13,46,50,23,53,43,
       5,48,40,48,10,31,15,35,50,8,48,55,52,18,54,16,35,4,43,55,34,13,5,13,27,41,19,22,21,26,48,4,15,1,45,51,13,49,22,
33,18,18,52,27,6,41,7,11,48,17,37,31,42,3,45,22,6,45,42,5,28,39,35,30,24,21,49,49,47,54,28,42,40,26,47,8,28,1,44,4,45,23,49,53,12,48,16,27,36,21,18,41,43,9,55,27,37,41,5,43,12,45,0,34,19,48,14,22,43,14,13,38,15,7,41,8,37,13,45,31,47,38,45,38,50,44,20,40,39,38,26,29,24,10,30,23,53,38,39,3,37,4,15,22,29,4,5,4,4,19,35,30,30,49,16,32,36,26,37,53,46,28,24,13,12,29,2,36,21,19,15,11,22,10,30,29,40,14,17,39,36,17,23,39,13,29,51,8,55,10,10,47,39,1,46,27,18,7,49,38,27,14,26,35,5,46,54,12,18,30,11,6,29,52,44,38,51,22,26,24,41,13,39,27,0,36,38,7,37,7,9,25,4,8,52,33,46,33,42,43,17,23,20,23,41,8,47,16,48,46,35,35,24,0,17,12,40,52,11,16,7,33,6,21,30,32,55,52,52,28,11,35,39,15,27,47,52,0,11,41,50,10,13,10,5,40,21,0,27,12,39,20,27,39,19,28,5,51,45,19,3,1,15,53,31,45,36,33,22,4,22,20,30,7,54,13,19,48,32,13,38,9,4,22,26,22,43,5,47,14,15,21,15,48,10,15,47,22,9,52,4,16,22,47,9,13,44,15,5,19,2,55,36,49,25,52,21,5,19,46,2,51,45,12,54,47,47,23,17,26,52,36,49,4,42,18,20,22,47,18,37,28,19,11,21,13,37,51,2,43,36,43,0,40,50,27,40,41,31,41,53,2,17,4,35,52,22,25,21,16,5,41,54,5,40,3,38,12,10,53,48,28,9,7,46,28,3,9,44,29,39,3,41,15,29,30,47,21,12,27,53,7,34,53,53,50,53,23,46,52,40,21,3,46,49,17,14,36,10,54,51,22,25,34,49,49,26,45,45,17,30,19,6,49,41,9,19,25,51,23,55,11,0,1,13,26,15,41,38,19,51,21,11,8,55,25,21,15,18,15,48,4,37,36,5,45,13,13,16,26,0,41,8,51,20,38,28,46,3,37,52,6,24,11,18,39,25,48,16,41,50,44,48,7,29,14,15,28,7,46,15,16,46,20,35,6,27,10,17,1,42,30,37,41,37,52,55,4,6,17,27,33,45,37,11,29,16,42,33,39,35,33,30,25,37,6,25,22,43,31,42,25,15,27,19,15,10,1,45,24,23,17,17,33,12,53,30,44,30,2,30,8,45,49,48,46,52,13,41,29,46,39,50,51,19,55,36,29,23,8,11,49,43,45,51,2,41,53,6,38,52,26,45,0,55,14,42,25,10,4,39,5,47,42,55,37,35,53,14,27,54,41,19,34,38,43,22,41,23,12,13,17,36,23,41,15,34,24,44,21,3,25,10,14,43,16,35,45,1,22,26,20,5,8,37,42,47,1,2,33,2,21,8,36,50,6,45,40,48,16,12,1,55,0,53,51,35,19,14,3,21,33,2,54,33,2,18,53,34,19,23,32,27,39,13,48,27,12,7,50,16,49,35,27,12,29,6,34,7,18,50,6,49,24,1,18,53,27,36,0,37,42,51,12,38,8,29,38,11,24,4,2,15,19,1,25,9,30,50,30,8,15,50,52,25,8,29,44,7,41,30,45,9,23,20,49,15,16,48,34,10,44,52,22,4,6,2,8,23,13,39,7,31,29,4,35,43,2,33,34,24,41,38,47,5,38,28,34,1,44,14,13,12,8,6,0,24,9,4,39,37,32,18,38,15,29,14,11,40,34,12,16,7,7,50,18,24,30,15,13,41,42,47,22,17,14,38,46,45,22,12,12,12,25,11,24,41,8,51,51,25,46,15,13,47,38,20,0,21,7,30,49,42,43,53,24,19,17,46,50,15,33,1,42,14,55,26,17,41,39,23,46,35,39,16,16,37,50,38,24,20,32,51,22,53,50,10,39,4,2,27,1,25,18,40,50,9,35,37,27,37,39,29,2,38,32,6,30,32,4,43,46,21,9,40,45,49,34,37,4,55,19,47,3,42,33,13,43,3,3,22,49]
#nums = [2,2,3,4]
#nums = [4,2,3,4]
#nums = [1,2,3]
#nums = [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9]
#nums = [400,400,400,400,400]

#triangleNumber(nums)

r'''
 双指针 977、344、2108、2000、151、524、283、611、80、986
        167\16.24\15\18\1471\11\1963\1679\16\offer007
'''
# 输入：nums = [0,0,1,1,1,1,2,3,3]
# 输出：7, nums = [0,0,1,1,2,3,3]
#歌华、陕西知识库
#   80
def removeDuplicates(nums):
    start, end = 0,0
    if len(nums)<=2:
        return nums
    tmp = 2
    while end < len(nums)-1:

        if nums[end+1] == nums[end]:
            tmp -= 1
        else:
            tmp = 2

        if tmp > 0:
            start += 1


        end += 1
        nums[start] = nums[end]

    print('start+1->',start+1)
    return nums[:start+1]

nums = [0,0,1,1,1,1,2,3,3]
#nums = [0,0,1,1,1,1,2,2,2,2,3,4,4,4]
nums = [1,1,1,2,2,3]
nums = [1]
#print(removeDuplicates(nums))

#   986
#firstList = [[0,2],[5,10],[13,23],[24,25]]
# secondList = [[1,5],[8,12],[15,24],[25,26]]
#[[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
def intervalIntersection(firstList, secondList):
    #size = min(len(firstList), len(secondList))
    res = []
    first, second = 0, 0
    # if len(firstList) ==0 or len(secondList)==0:
    #     return []
    while first < len(firstList) and second < len(secondList):
        print(first, second)
        first1, first2 = firstList[first]
        second1, second2 = secondList[second]
        if max(first1, second1) <= min(first2, second2):
            res.append([max(first1, second1), min(first2, second2)])
        if  first2 < second2:
            first += 1
        else:
            second += 1


    return  res
firstList = [[0,2],[5,10],[13,23],[24,25]]
secondList = [[1,5],[8,12],[15,24],[25,26]]

firstList = [[1,3],[5,9]]
secondList = []
#
firstList = []
secondList = [[4,8],[10,12]]
#
firstList = [[1,5],[6,10],[13,23],[24,25]]
secondList = [[0,2],[8,12],[15,24],[25,26]]

firstList = [[8,15]]
secondList = [[2,6],[8,10],[12,20]]
#
# firstList = [[2,6],[8,10],[12,20]]
# secondList = [[8,15]]

firstList = [[0,5],[12,14],[15,18]]
secondList = [[11,15],[12,13],[18,19]]
#print(intervalIntersection(firstList, secondList))

#   interview 16.24
def pairSums(nums, target):
    nums.sort()
    res = []
    start,end = 0, len(nums)-1
    while start < end:
        # if start > 0 and nums[start] == nums[start-1]:
        #     start += 1
        # if end < len(nums)-1 and nums[end] == nums[end+1]:
        #     end -= 1
        if nums[start]+nums[end] == target:
            res.append([nums[start],nums[end]])
            start += 1
            end -= 1
        elif nums[start]+nums[end] > target:
            end -= 1
        else:
            start += 1

    return res

nums = [5,6,5]
nums = [5]
target = 11
#print(pairSums(nums, target))

#1471
#[1,2,3,4,5], k = 2
def getStrongest(arr, k):
    arr.sort()
    start, end = 0, len(arr)-1
    res = []
    while start <= end and k > 0:
        #print(arr[(len(arr)-1)//2])
        if abs(arr[start] - arr[(len(arr)-1)//2]) > abs(arr[end] - arr[(len(arr)-1)//2]):
            res.append(arr[start])
            k -=1
            start += 1
        else:
            res.append(arr[end])
            k -= 1
            end -= 1

    return res

arr = [1,2,3,4,5]
k = 2

# arr = [1,1,3,5,5]
# k = 2

# arr = [6,7,11,7,6,8]
# k = 5

# arr = [6,-3,7,2,11]
# k = 3
arr = [-7,22,17,3]
k = 2
# arr = [-1]
# k = 2
#print(getStrongest(arr, k))
#11
# 输入：[1,8,6,2,5,4,8,3,7]
# 输出：49
def maxArea(height):

#1963
#1679
#16