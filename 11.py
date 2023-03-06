# 输入：[1,8,6,2,5,4,8,3,7]
# 输出：49

#双指针，小的先走
def maxArea(height):
    start, end = 0, len(height) - 1
    m = 0
    while start <= end:
        m = max(m, min(height[start], height[end]) * (end - start))
        if height[start] < height[end]:

            start += 1
        else:

            end -= 1
    return m