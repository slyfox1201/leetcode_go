package main

import (
	"container/heap"
	"container/list"
	"fmt"
	"math"
	"math/bits"
	"sort"
	"strconv"
	"strings"
)

// 1
func twoSum(nums []int, target int) []int {
	cache := make(map[int]int)
	for i, j := range nums {
		if n, ok := cache[target-j]; ok {
			return []int{n, i}
		}
		cache[j] = i
	}
	return nil
}

// 2
type ListNode struct {
	Val  int
	Next *ListNode
}

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	head := new(ListNode)
	tail := head
	carry := 0
	for l1 != nil || l2 != nil {
		n1, n2 := 0, 0
		if l1 != nil {
			n1 = l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			n2 = l2.Val
			l2 = l2.Next
		}
		sum := n1 + n2 + carry
		sum, carry = sum%10, sum/10
		tail.Next = &ListNode{Val: sum}
		tail = tail.Next
	}
	if carry > 0 {
		tail.Next = &ListNode{Val: carry}
	}
	return head.Next
}

// 3
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

func lengthOfLongestSubstring(s string) int {
	set := make(map[byte]bool, 0)
	ans := 0
	for lp, rp := 0, 0; lp < len(s); lp++ {
		if lp != 0 {
			delete(set, s[lp-1])
		}
		for ; rp < len(s) && !set[s[rp]]; rp++ {
			set[s[rp]] = true
		}
		ans = max(ans, rp-lp)
	}
	return ans
}

// 4
func findMedianSortedArrays1(nums1 []int, nums2 []int) float64 {
	m, n := len(nums1), len(nums2)
	nums := make([]int, m+n)
	i, j := 0, 0
	var ans float64
	nl := m + n
	for k := 0; k < m+n; k++ {
		if i < m && j < n {
			if nums1[i] < nums2[j] {
				nums[k] = nums1[i]
				i++
			} else {
				nums[k] = nums2[j]
				j++
			}
		} else if i < m {
			nums[k] = nums1[i]
			i++
		} else {
			nums[k] = nums2[j]
			j++
		}
	}
	fmt.Println(nl, nums)
	if nl%2 == 0 {
		ans = float64(nums[nl/2]+nums[nl/2-1]) / 2
	} else {
		ans = float64(nums[nl/2])
	}
	return ans
}

func findMedianSortedArrays2(nums1 []int, nums2 []int) float64 {
	m, n := len(nums1), len(nums2)
	mn := m + n
	left, right := -1, -1
	i, j := 0, 0
	for k := 0; k <= mn/2; k++ {
		left = right
		if i < m && (j >= n || nums1[i] < nums2[j]) {
			right = nums1[i]
			i++
		} else {
			right = nums2[j]
			j++
		}
	}
	if mn&1 == 0 {
		return float64(left+right) / 2.0
	} else {
		return float64(right)
	}
}

func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	m, n := len(nums1), len(nums2)
	left, right := (m+n+1)/2, (m+n+2)/2
	return float64(getKth(nums1, 0, m-1, nums2, 0, n-1, left)+getKth(nums1, 0, m-1, nums2, 0, n-1, right)) / 2
}

func getKth(nums1 []int, start1 int, end1 int, nums2 []int, start2 int, end2 int, k int) int {
	len1 := end1 - start1 + 1
	len2 := end2 - start2 + 1
	if len1 > len2 {
		return getKth(nums2, start2, end2, nums1, start1, end1, k)
	}
	if len1 == 0 {
		return nums2[start2+k-1]
	}
	if k == 1 {
		return min(nums1[start1], nums2[start2])
	}
	i := start1 + min(len1, k/2) - 1
	j := start2 + min(len2, k/2) - 1
	if nums1[i] > nums2[j] {
		return getKth(nums1, start1, end1, nums2, j+1, end2, k-(j-start2+1))
	} else {
		return getKth(nums1, i+1, end1, nums2, start2, end2, k-(i-start1+1))
	}
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

// 5
func longestPalindrome1(s string) string {
	start, end := 0, 0
	for i := range s {
		left1, right1 := helper5(s, i, i)
		left2, right2 := helper5(s, i, i+1)
		if right1-left1 > end-start {
			start, end = left1, right1
		}
		if right2-left2 > end-start {
			start, end = left2, right2
		}
	}
	return s[start : end+1]
}
func helper5(s string, left, right int) (int, int) {
	for ; left >= 0 && right < len(s) && s[left] == s[right]; left, right = left-1, right+1 {
	}
	return left + 1, right - 1
}

func longestPalindrome(s string) string {
	length := len(s)
	if length < 2 {
		return s
	}
	maxLen := 1
	begin := 0
	dp := make([][]bool, length)
	for i := range dp {
		dp[i] = make([]bool, length)
	}
	for i := 0; i < length; i++ {
		dp[i][i] = true
	}
	for L := 2; L <= length; L++ {
		for i := 0; i < length; i++ {
			j := L + i - 1
			if j >= length {
				break
			}
			if s[i] != s[j] {
				dp[i][j] = false
			} else {
				if j-i < 3 {
					dp[i][j] = true
				} else {
					dp[i][j] = dp[i+1][j-1]
				}
			}
			if dp[i][j] && j-i+1 > maxLen {
				maxLen = j - i + 1
				begin = i
			}
		}
	}
	return s[begin : begin+maxLen]
}

// 6
func convert1(s string, numRows int) string {
	if numRows == 1 {
		return s
	}
	cache := make([][]byte, numRows)
	for i := range cache {
		cache[i] = make([]byte, 0)
	}
	cur := 0
	p := -1
	for i := range s {
		if i%(numRows-1) == 0 {
			p = -p
		}
		cache[cur] = append(cache[cur], s[i])
		cur += p
	}
	ans := ""
	for _, j := range cache {
		s2 := string(j)
		ans += s2
	}
	return ans
}

func convert(s string, numRows int) string {
	if numRows == 1 {
		return s
	}
	n := 2*numRows - 2
	rows := make([]string, numRows)
	for i, c := range s {
		x := i % n
		rows[min(x, n-x)] += string(c)
	}
	return strings.Join(rows, "")
}

// 7
func reverse(x int) int {
	ans := 0
	for x != 0 {
		ans *= 10
		ans += x % 10
		x /= 10
	}
	if ans >= math.MaxInt32 || ans <= math.MinInt32 {
		return 0
	}
	return ans
}

// 8
// 状态表
var table = [][]int{
	{0, 1, 2, 3},
	{3, 3, 2, 3},
	{3, 3, 2, 3},
	//{3, 3, 3, 3},
}

func myAtoi(str string) int {
	currentState := 0
	sign := 1
	res := 0
LOOP:
	for i := 0; i < len(str); i++ {
		currentState = table[currentState][getType(str[i])]
		switch currentState {
		case 0:
		case 1:
			if str[i] == '-' {
				sign = -1
			} else {
				sign = 1
			}
		case 2:
			res = res*10 + int(str[i]-'0')
			if withSign := res * sign; withSign > math.MaxInt32 {
				return math.MaxInt32
			} else if withSign < math.MinInt32 {
				return math.MinInt32
			}
		case 3:
			break LOOP
		}
	}
	return res * sign
}

func getType(c byte) int {
	switch {
	case c == ' ':
		return 0
	case c == '+' || c == '-':
		return 1
	case c >= '0' && c <= '9':
		return 2
	default:
		return 3
	}
}

// 9
func isPalindrome(x int) bool {
	if x < 0 {
		return false
	}
	y := 0
	z := x
	for x != 0 {
		y *= 10
		y += x % 10
		x /= 10
	}
	return z == y
}

// 10
func isMatch(s string, p string) bool {
	m, n := len(s), len(p)
	dp := make([][]bool, m+1)
	for i := range dp {
		dp[i] = make([]bool, n+1)
	}
	dp[0][0] = true
	for i := 0; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if p[j-1] == '*' {
				dp[i][j] = dp[i][j-2] // a*不进行匹配
				if matches(s, p, i, j-1) {
					dp[i][j] = dp[i][j] || dp[i-1][j] // a*继续匹配
				}
			} else {
				if matches(s, p, i, j) {
					dp[i][j] = dp[i-1][j-1]
				}
			}
		}
	}
	return dp[m][n]
}

func matches(s, p string, i, j int) bool {
	if i == 0 {
		return false
	}
	if p[j-1] == '.' {
		return true
	}
	return s[i-1] == p[j-1]
}

// 11
func maxArea(height []int) int {
	i, j := 0, len(height)-1
	ans := 0
	for i < j {
		tmp := 0
		if height[i] < height[j] {
			tmp = height[i] * (j - i)
			i++
		} else {
			tmp = height[j] * (j - i)
			j--
		}
		ans = max11(ans, tmp)
	}
	return ans
}

func max11(x, y int) int {
	if x > y {
		return x
	}
	return y
}

// 12
func intToRoman(num int) string {
	values := []int{1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1}
	symbols := []string{"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"}
	ans := make([]byte, 0)
	for i := 0; i < len(values); i++ {
		for num >= values[i] {
			num -= values[i]
			ans = append(ans, symbols[i]...)
		}
	}
	return string(ans)
}

func romanToInt(s string) int {
	cache := map[byte]int{
		'M': 1000,
		'D': 500,
		'C': 100,
		'L': 50,
		'X': 10,
		'V': 5,
		'I': 1,
	}
	ans := 0
	for i := range s {
		if i != len(s)-1 && cache[s[i]] < cache[s[i+1]] {
			ans -= cache[s[i]]
		} else {
			ans += cache[s[i]]
		}
	}
	return ans
}

// 14
func longestCommonPrefix1(strs []string) string {
	shortest := strs[0]
	for _, str := range strs {
		if len(str) < len(shortest) {
			shortest = str
		}
	}
	i := 0
LOOP:
	for ; i < len(shortest); i++ {
		for _, str := range strs {
			if str[i] != shortest[i] {
				break LOOP
			}
		}
	}
	return shortest[:i]
}

func longestCommonPrefix(strs []string) string {
	if len(strs) == 0 {
		return ""
	}
	for i := 0; i < len(strs[0]); i++ {
		for j := 1; j < len(strs); j++ {
			if i == len(strs[j]) || strs[j][i] != strs[0][i] {
				return strs[0][:i]
			}
		}
	}
	return strs[0]
}

// 15
func threeSum(nums []int) [][]int {
	ans := make([][]int, 0)
	n := len(nums)
	sort.Ints(nums)
	for k := 0; k < n; k++ {
		if k > 0 && nums[k] == nums[k-1] {
			continue
		}
		i, j := k+1, n-1
		for i < j {
			sum := nums[i] + nums[j] + nums[k]
			if sum == 0 {
				ans = append(ans, []int{nums[i], nums[j], nums[k]})
				for i < j && nums[i] == nums[i+1] {
					i++
				}
				for i < j && nums[j-1] == nums[j] {
					j--
				}
				i++
				j--
			} else if sum > 0 {
				j--
			} else if sum < 0 {
				i++
			}
		}
	}
	return ans
}

// 16
func threeSumClosest(nums []int, target int) int {
	sort.Ints(nums)
	ans := nums[0] + nums[1] + nums[2]
	n := len(nums)
	for k := 0; k < n; k++ {
		i, j := k+1, n-1
		for i < j {
			sum := nums[i] + nums[j] + nums[k]
			delta := sum - target
			if abs(delta) < ans {
				ans = abs(delta)
			}
			if delta == 0 {
				return ans
			} else if delta > 0 {
				j--
			} else if delta < 0 {
				i++
			}
		}
	}
	return ans
}
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// 17
func letterCombinations(digits string) []string {
	if len(digits) == 0 {
		return nil
	}
	ans := make([]string, 0)
	phoneMap := map[string]string{
		"2": "abc",
		"3": "def",
		"4": "ghi",
		"5": "jkl",
		"6": "mno",
		"7": "pqrs",
		"8": "tuv",
		"9": "wxyz",
	}
	var helper func(index int, cur string)
	helper = func(index int, cur string) {
		if index == len(digits) {
			ans = append(ans, cur)
			return
		}
		letters := phoneMap[string(digits[index])]
		for i := 0; i < len(letters); i++ {
			helper(index+1, cur+string(letters[i]))
		}
	}
	helper(0, "")
	return ans
}

// 18
func fourSum(nums []int, target int) [][]int {
	ans := make([][]int, 0)
	sort.Ints(nums)
	n := len(nums)
	for l := 0; l < n; l++ {
		if l > 0 && nums[l] == nums[l-1] {
			continue
		}
		for r := n - 1; 0 < r; r-- {
			if r < n-1 && nums[r] == nums[r+1] {
				continue
			}
			i, j := l+1, r-1
			for i < j {
				sum := nums[i] + nums[j] + nums[l] + nums[r]
				if sum == target {
					ans = append(ans, []int{nums[i], nums[j], nums[l], nums[r]})
					for i < j && nums[i] == nums[i+1] {
						i++
					}
					for i < j && nums[j-1] == nums[j] {
						j--
					}
					i++
					j--
				} else if sum > target {
					j--
				} else if sum < target {
					i++
				}
			}
		}
	}
	return ans
}

// 19
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummy := new(ListNode)
	dummy.Next = head
	q := dummy
	p := dummy
	for i := 0; i < n; i++ {
		p = p.Next
	}
	for p.Next != nil {
		q = q.Next
		p = p.Next
	}
	q.Next = q.Next.Next
	return dummy.Next
}

// 20
func isValid1(s string) bool {
	l := list.New()
	for _, c := range s {
		if c == '(' || c == '{' || c == '[' {
			l.PushBack(c)
		} else {
			if l.Len() == 0 {
				return false
			}
			switch c {
			case ')':
				remove := l.Remove(l.Back())
				if remove != '(' { // 和常量的对比不需要使用类型断言
					return false
				}
			case ']':
				remove := l.Remove(l.Back())
				if remove != '[' {
					return false
				}
			case '}':
				remove := l.Remove(l.Back())
				if remove != '{' {
					return false
				}
			}
		}
	}
	return l.Len() == 0
}

// 使用 container/list 模拟栈
func isValid2(s string) bool {
	l := list.New()
	getLeft := func(c byte) byte {
		switch c {
		case ')':
			return '('
		case ']':
			return '['
		case '}':
			return '{'
		}
		return ' '
	}
	for _, c := range s {
		if c == '(' || c == '{' || c == '[' {
			l.PushBack(byte(c))
		} else {
			if l.Len() == 0 || getLeft(byte(c)) != l.Remove(l.Back()).(byte) { // 和变量的对比需要使用类型断言
				return false
			}
		}
	}
	return l.Len() == 0
}

// 使用 数组 模拟栈
func isValid(s string) bool {
	stack := make([]byte, 0)
	getLeft := func(c byte) byte {
		switch c {
		case ')':
			return '('
		case ']':
			return '['
		case '}':
			return '{'
		}
		return ' '
	}
	for _, c := range s {
		if c == '(' || c == '{' || c == '[' {
			stack = append(stack, byte(c))
		} else {
			if len(stack) == 0 || getLeft(byte(c)) != stack[len(stack)-1] {
				return false
			}
			stack = stack[:len(stack)-1]
		}
	}
	return len(stack) == 0
}

// 21
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	dummy := new(ListNode)
	p := dummy
	for l1 != nil && l2 != nil {
		if l1.Val < l2.Val {
			p.Next = l1
			l1 = l1.Next
		} else {
			p.Next = l2
			l2 = l2.Next
		}
		p = p.Next
	}
	if l1 != nil {
		p.Next = l1
	}
	if l2 != nil {
		p.Next = l2
	}
	return dummy.Next
}

// 22
func generateParenthesis(n int) []string {
	ans := make([]string, 0)
	var dfs func(left int, right int, cur []byte)
	dfs = func(left int, right int, cur []byte) {
		if left < 0 || right < 0 || left > right {
			return
		}
		if left == 0 && right == 0 {
			ans = append(ans, string(cur))
			return
		}
		cur = append(cur, '(')
		dfs(left-1, right, cur)
		cur = cur[:len(cur)-1]

		cur = append(cur, ')')
		dfs(left, right-1, cur)
		cur = cur[:len(cur)-1] // wjq 注：此句没有意义，可以删掉
	}
	dfs(n, n, []byte{})
	return ans
}

// wjq 回溯杂谈
// dfs是记录当前结果的cur，一般是可以直接使用切片，而不需要使用切片指针。因为cur的作用
// 是用来向深层递归传递信息，一般情况下，它的长度是线性增加和线性减少，增加的次数和减少的
// 次数是相同的。在append时创建新的底层数组，并不会对逻辑产生影响。换句话说，如果使用
// 切片指针来记录信息，底层数组永远只有一个，尚且不会出现问题，使用普通切片，只会让深层
// 递归创建新的数组，并不会影响当前层的逻辑，只是在创建新数组时，多了额外的开销。因此，从
// 性能角度来看，使用切片指针更优。但普通切片会更方便，刷题足矣。

// 23
func mergeKLists(lists []*ListNode) *ListNode {
	dummy := new(ListNode)
	p := dummy
	pq := &PriorityQueue{}
	heap.Init(pq)
	for _, l := range lists {
		if l != nil {
			heap.Push(pq, l)
		}
	}
	for pq.Len() > 0 {
		node := heap.Pop(pq).(*ListNode)
		if node.Next != nil {
			heap.Push(pq, node.Next)
		}
		p.Next = node
		p = p.Next
	}
	return dummy.Next
}

type PriorityQueue []*ListNode

func (p PriorityQueue) Len() int {
	return len(p)
}

func (p PriorityQueue) Less(i, j int) bool {
	return p[i].Val < p[j].Val
}

func (p PriorityQueue) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

func (p *PriorityQueue) Push(x interface{}) {
	*p = append(*p, x.(*ListNode))
}

func (p *PriorityQueue) Pop() interface{} {
	old := *p
	n := len(old)
	x := old[n-1]
	*p = old[:n-1]
	return x
}

// 24
func swapPairs(head *ListNode) *ListNode {
	return reverseKGroup(head, 2)
}

// 25
func reverseKGroup(head *ListNode, k int) *ListNode {
	dummy := new(ListNode)
	dummy.Next = head
	pre := dummy
	end := dummy
	reverse := func(head *ListNode) *ListNode {
		pre := new(ListNode)
		curr := head
		for curr != nil {
			next := curr.Next
			curr.Next = pre
			pre = curr
			curr = next
		}
		return pre
	}
	for end.Next != nil {
		for i := 0; i < k && end != nil; i++ {
			end = end.Next
		}
		if end == nil {
			break
		}
		start := pre.Next
		next := end.Next
		end.Next = nil
		pre.Next = reverse(start)
		start.Next = next
		pre = start
		end = pre
	}
	return dummy.Next
}

// 26
func removeDuplicates(nums []int) int {
	slow := 0
	for fast := 0; fast < len(nums); fast++ {
		if fast == 0 || nums[fast] != nums[fast-1] {
			nums[slow] = nums[fast]
			slow++
		}
	}
	return slow
}

//27
func removeElement(nums []int, val int) int {
	slow := 0
	for fast := 0; fast < len(nums); fast++ {
		if nums[fast] != val {
			nums[slow] = nums[fast]
			slow++
		}
	}
	return slow
}

// wjq: todo
// 28 KMP
func strStr(haystack, needle string) int {
	n, m := len(haystack), len(needle)
	if m == 0 {
		return 0
	}
	pi := make([]int, m)
	for i, j := 1, 0; i < m; i++ {
		for j > 0 && needle[i] != needle[j] {
			j = pi[j-1]
		}
		if needle[i] == needle[j] {
			j++
		}
		pi[i] = j
	}
	for i, j := 0, 0; i < n; i++ {
		for j > 0 && haystack[i] != needle[j] {
			j = pi[j-1]
		}
		if haystack[i] == needle[j] {
			j++
		}
		if j == m {
			return i - m + 1
		}
	}
	return -1
}

// 29
func divide(dividend int, divisor int) int {
	if divisor == -1 && dividend == math.MinInt32 {
		return math.MaxInt32
	}
	var helper func(dividend int, divisor int) int
	helper = func(dividend int, divisor int) int {
		if dividend < divisor {
			return 0
		}
		sum, count := divisor, 1
		for dividend >= sum {
			sum <<= 1
			count <<= 1
		}
		sum >>= 1
		count >>= 1
		return count + helper(dividend-sum, divisor)
	}
	abs := func(x int) int {
		if x < 0 {
			return -x
		}
		return x
	}
	if dividend > 0 && divisor > 0 || dividend < 0 && divisor < 0 {
		return helper(abs(dividend), abs(divisor))
	} else {
		return -helper(abs(dividend), abs(divisor))
	}
}

// 30
func findSubstring(s string, words []string) []int {
	if len(words) == 0 {
		return nil
	}
	ans := make([]int, 0)
	sLen, wLen := len(s), len(words)
	size := len(words[0])
	mp := make(map[string]int, 0)
	for _, w := range words {
		mp[w]++
	}
	for i := 0; i < size; i++ {
		curMp := make(map[string]int, 0)
		cnt := 0
		for l, r := i, i; r <= sLen-size; r += size {
			word := s[r : r+size]
			if num, ok := mp[word]; ok {
				for curMp[word] >= num {
					curMp[s[l:l+size]]--
					cnt--
					l += size
				}
				curMp[word]++
				cnt++
			} else {
				for l < r {
					curMp[s[l:l+size]]--
					cnt--
					l += size
				}
				l += size
			}
			if cnt == wLen {
				ans = append(ans, l)
			}
		}
	}
	return ans
}

// 31
func nextPermutation(nums []int) {
	reverse := func(a []int) {
		for i, n := 0, len(a); i < n/2; i++ {
			a[i], a[n-1-i] = a[n-1-i], a[i]
		}
	}
	n := len(nums)
	i := n - 2
	for i >= 0 && nums[i] >= nums[i+1] {
		i--
	}
	if i >= 0 {
		j := n - 1
		for j > 0 && nums[i] >= nums[j] {
			j--
		}
		nums[i], nums[j] = nums[j], nums[i]
	}
	reverse(nums[i+1:])
}

// 32
func longestValidParentheses(s string) int {
	left, right := 0, 0
	ans := 0
	max := func(x, y int) int {
		if x > y {
			return x
		}
		return y
	}
	for i := 0; i < len(s); i++ {
		if s[i] == '(' {
			left++
		} else {
			right++
		}
		if left == right {
			ans = max(ans, left*2)
		} else if left < right {
			left, right = 0, 0
		}
	}
	left, right = 0, 0
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == '(' {
			left++
		} else {
			right++
		}
		if left == right {
			ans = max(ans, left*2)
		} else if left > right {
			left, right = 0, 0
		}
	}
	return ans
}

func longestValidParentheses1(s string) int { // DP解法
	ans := 0
	dp := make([]int, len(s))
	max := func(x, y int) int {
		if x > y {
			return x
		}
		return y
	}
	for i := 1; i < len(s); i++ {
		if s[i] == ')' {
			if s[i-1] == '(' {
				if i-2 >= 0 {
					dp[i] = dp[i-2] + 2
				} else {
					dp[i] = 2
				}
			} else if i-dp[i-1]-1 >= 0 && s[i-dp[i-1]-1] == '(' {
				if i-dp[i-1]-2 >= 0 {
					dp[i] = dp[i-1] + 2 + dp[i-dp[i-1]-2]
				} else {
					dp[i] = dp[i-1] + 2
				}
			}
			ans = max(ans, dp[i])
		}
	}
	return ans
}

// 33
func search(nums []int, target int) int {
	n := len(nums)
	left, right := 0, n
	for left < right {
		mid := (left + right) >> 1
		if nums[mid] == target {
			return mid
		}
		if nums[0] < nums[mid] {
			if nums[0] <= target && target < nums[mid] {
				right = mid
			} else {
				left = mid + 1
			}
		} else {
			if nums[mid] < target && target <= nums[right-1] {
				left = mid + 1
			} else {
				right = mid
			}
		}
	}
	return -1
}

// 34
func searchRange(nums []int, target int) []int {
	helper := func(nums []int, target int) int {
		left, right := 0, len(nums)
		for left < right {
			mid := (left + right) >> 1
			if nums[mid] <= target {
				left = mid + 1
			} else if nums[mid] > target {
				right = mid
			}
		}
		return left
	}
	right := helper(nums, target)
	if right-1 < len(nums) && right-1 >= 0 && nums[right-1] == target {
		return []int{helper(nums, target-1), right - 1}
	} else {
		return []int{-1, -1}
	}
}

func searchRange1(nums []int, target int) []int { // 使用包函数进行二分搜索
	leftmost := sort.SearchInts(nums, target)
	if leftmost == len(nums) || nums[leftmost] != target {
		return []int{-1, -1}
	}
	rightmost := sort.SearchInts(nums, target+1) - 1
	return []int{leftmost, rightmost}
}

// 35
func searchInsert(nums []int, target int) int {
	left, right := 0, len(nums)
	for left < right {
		mid := (left + right) >> 1
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			left = mid + 1
		} else if nums[mid] > target {
			right = mid
		}
	}
	return left
}

func searchInsert1(nums []int, target int) int {
	left, right := 0, len(nums)
	for left < right {
		mid := (left + right) >> 1
		if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid
		}
	}
	return left
}

// 36
func isValidSudoku(board [][]byte) bool { // 位图太美了~~~
	var row, col, block [9]uint16
	var cur uint16
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			if board[i][j] == '.' {
				continue
			}
			bid := i/3*3 + j/3
			cur = 1 << (board[i][j] - '1')
			if (cur&row[i])|(cur&col[j])|(cur&block[bid]) != 0 {
				return false
			}
			row[i] |= cur
			col[j] |= cur
			block[bid] |= cur
		}
	}
	return true
}

// 37
func solveSudoku(board [][]byte) {
	var row, col, block [9]int
	var spaces [][2]int

	flip := func(i, j int, digit byte) {
		row[i] ^= 1 << digit
		col[j] ^= 1 << digit
		block[i/3*3+j/3] ^= 1 << digit
	}

	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			if board[i][j] == '.' {
				spaces = append(spaces, [2]int{i, j})
			} else {
				digit := board[i][j] - '1'
				flip(i, j, digit)
			}
		}
	}

	var dfs func(int) bool
	dfs = func(pos int) bool {
		if pos == len(spaces) {
			return true
		}
		i, j := spaces[pos][0], spaces[pos][1]
		mask := 0x1ff &^ uint(row[i]|col[j]|block[i/3*3+j/3])
		for ; mask > 0; mask &= mask - 1 {
			digit := byte(bits.TrailingZeros(mask)) // 返回二进制中最后的1的索引（最右索引为0，向左递增）
			flip(i, j, digit)
			board[i][j] = digit + '1'
			if dfs(pos + 1) {
				return true
			}
			flip(i, j, digit)
		}
		return false
	}
	dfs(0)
}

// 38
func countAndSay(n int) string {
	ans := []byte{'1', '$'}
	for i := 1; i < n; i++ {
		var cnt byte = 1
		tmp := make([]byte, 0)
		for j := 1; j < len(ans); j++ {
			if ans[j] == ans[j-1] {
				cnt++
			} else {
				tmp = append(tmp, cnt+'0', ans[j-1])
				cnt = 1
			}
		}
		tmp = append(tmp, '$')
		ans = tmp
	}
	return string(ans[:len(ans)-1])
}

// 39
func combinationSum(candidates []int, target int) [][]int {
	ans := make([][]int, 0)
	var dfs func(cur []int, sum int, start int)
	dfs = func(cur []int, sum int, start int) {
		if sum >= target {
			if sum == target {
				tmp := make([]int, len(cur))
				copy(tmp, cur)
				ans = append(ans, tmp)
			}
			return
		}
		for i := start; i < len(candidates); i++ {
			cur = append(cur, candidates[i])
			dfs(cur, sum+candidates[i], i)
			cur = cur[:len(cur)-1]
		}
	}
	dfs([]int{}, 0, 0)
	return ans
}

// 40
func combinationSum2(candidates []int, target int) [][]int {
	ans := make([][]int, 0)
	sort.Ints(candidates)
	var dfs func(cur []int, sum int, start int)
	dfs = func(cur []int, sum int, start int) {
		if sum >= target {
			if sum == target {
				tmp := make([]int, len(cur))
				copy(tmp, cur)
				ans = append(ans, tmp)
			}
			return
		}
		for i := start; i < len(candidates); i++ {
			if i != start && candidates[i] == candidates[i-1] {
				continue
			}
			cur = append(cur, candidates[i])
			dfs(cur, sum+candidates[i], i+1)
			cur = cur[:len(cur)-1]
		}
	}
	dfs([]int{}, 0, 0)
	return ans
}

// 41
func firstMissingPositive(nums []int) int {
	n := len(nums)
	for i := 0; i < n; i++ {
		for nums[i] > 0 && nums[i] <= n && nums[nums[i]-1] != nums[i] {
			nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
		}
	}
	for i := 0; i < n; i++ {
		if nums[i] != i+1 {
			return i + 1
		}
	}
	return n + 1
}

// 42
func trap1(height []int) int {
	if len(height) == 0 {
		return 0
	}
	l, r := 0, len(height)-1
	lmax, rmax := height[l], height[r]
	ans := 0
	for l < r {
		if height[l] < height[r] {
			if height[l] < lmax {
				ans += lmax - height[l]
			} else {
				lmax = height[l]
			}
			l++
		} else {
			if height[r] < rmax {
				ans += rmax - height[r]
			} else {
				rmax = height[r]
			}
			r--
		}
	}
	return ans
}

func trap(height []int) (ans int) {
	left, right := 0, len(height)-1
	leftMax, rightMax := 0, 0
	max := func(a, b int) int {
		if a > b {
			return a
		}
		return b
	}
	for left < right {
		leftMax = max(leftMax, height[left])
		rightMax = max(rightMax, height[right])
		if height[left] < height[right] {
			ans += leftMax - height[left]
			left++
		} else {
			ans += rightMax - height[right]
			right--
		}
	}
	return
}

// 43
func multiply1(num1 string, num2 string) string {
	if num1 == "0" || num2 == "0" {
		return "0"
	}
	m, n := len(num1), len(num2)
	ansArr := make([]int, m+n)
	for i := m - 1; i >= 0; i-- {
		x := int(num1[i]) - '0'
		for j := n - 1; j >= 0; j-- {
			y := int(num2[j] - '0')
			ansArr[i+j+1] += x * y
		}
	}
	for i := m + n - 1; i > 0; i-- {
		ansArr[i-1] += ansArr[i] / 10
		ansArr[i] %= 10
	}
	ans := ""
	idx := 0
	if ansArr[0] == 0 {
		idx = 1
	}
	//fmt.Println(ansArr)
	for ; idx < m+n; idx++ {
		ans += strconv.Itoa(ansArr[idx])
	}
	return ans
}

func multiply(num1 string, num2 string) string {
	if num1 == "0" || num2 == "0" {
		return "0"
	}
	m, n := len(num1), len(num2)
	arr := make([]int, m+n)
	for i := m - 1; i >= 0; i-- {
		x := int(num1[i]) - '0'
		for j := n - 1; j >= 0; j-- {
			y := int(num2[j]) - '0'
			arr[i+j+1] += x * y
		}
	}
	for i := m + n - 1; i > 0; i-- {
		arr[i-1] += arr[i] / 10
		arr[i] %= 10
	}
	index := 0
	ans := make([]byte, 0)
	if arr[0] == 0 {
		index = 1
	}
	for ; index < m+n; index++ {
		ans = append(ans, byte(arr[index])+'0')
	}
	return string(ans)
}

// 44
func isMatch44(s string, p string) bool {
	m, n := len(s), len(p)
	dp := make([][]bool, m+1)
	for i := range dp {
		dp[i] = make([]bool, n+1)
	}
	dp[0][0] = true
	for j := 1; j <= n; j++ {
		if p[j-1] == '*' {
			dp[0][j] = dp[0][j-1]
		} else {
			break
		}
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if p[j-1] == '*' {
				dp[i][j] = dp[i][j-1] || dp[i-1][j]
			} else if p[j-1] == '?' || p[j-1] == s[i-1] {
				dp[i][j] = dp[i-1][j-1]
			}
		}
	}
	return dp[m][n]
}

// 45
func jump(nums []int) int {
	ans := 0
	end := 0
	maxIndex := 0
	max := func(x, y int) int {
		if x > y {
			return x
		}
		return y
	}
	for i := 0; i < len(nums)-1; i++ {
		maxIndex = max(maxIndex, i+nums[i])
		if end == i {
			end = maxIndex
			ans++
		}
	}
	return ans
}

// 46
func permute(nums []int) [][]int {
	n := len(nums)
	ans := make([][]int, 0)
	used := make([]bool, n)
	var dfs func(depth int, cur []int)
	dfs = func(depth int, cur []int) {
		if depth == n {
			tmp := make([]int, len(cur))
			copy(tmp, cur)
			ans = append(ans, tmp)
			return
		}
		for i := 0; i < n; i++ {
			if !used[i] {
				cur = append(cur, nums[i])
				used[i] = true
				dfs(depth+1, cur)
				cur = cur[:len(cur)-1]
				used[i] = false
			}
		}
	}
	dfs(0, []int{})
	return ans
}

// 47
func permuteUnique(nums []int) [][]int {
	sort.Ints(nums)
	n := len(nums)
	ans := make([][]int, 0)
	used := make([]bool, n)
	var dfs func(depth int, cur []int)
	dfs = func(depth int, cur []int) {
		if depth == n {
			tmp := make([]int, len(cur))
			copy(tmp, cur)
			ans = append(ans, tmp)
			return
		}
		for i := 0; i < n; i++ {
			if i != 0 && nums[i] == nums[i-1] && !used[i-1] {
				continue
			}
			if !used[i] {
				cur = append(cur, nums[i])
				used[i] = true
				dfs(depth+1, cur)
				cur = cur[:len(cur)-1]
				used[i] = false
			}
		}
	}
	dfs(0, []int{})
	return ans
}

// 四个（N个）人过桥问题 https://zhuanlan.zhihu.com/p/163533221
func shortT(arr []int, n int) int {
	end := n
	times := 0
	if n == 1 {
		return arr[n-1]
	}
	if n == 2 {
		return arr[n-1]
	}
	if n == 3 {
		return arr[0] + arr[1] + arr[2]
	}
	if n >= 4 {
		a := arr[0]
		b := arr[1]
		y := arr[n-2]
		z := arr[n-1]
		if 2*a+z+y > 2*b+a+z {
			times = 2*b + a + z
			end = end - 2
		} else {
			times = 2*a + z + y
			end = end - 2
		}
		times = times + shortT(arr, end)
		return times
	}
	return 0
}

// 48
func rotate(matrix [][]int) {
	n := len(matrix)
	for i := 0; i < n/2; i++ {
		matrix[i], matrix[n-1-i] = matrix[n-1-i], matrix[i]
	}
	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
		}
	}
}

// 49
func groupAnagrams(strs []string) [][]string {
	ans := make([][]string, 0)
	mp := make(map[string][]string, 0)
	for _, str := range strs {
		s := []byte(str)
		sort.Slice(s, func(i, j int) bool {
			return s[i] < s[j]
		})
		tmp := string(s)
		mp[tmp] = append(mp[tmp], str)
	}
	for _, v := range mp {
		ans = append(ans, v)
	}
	return ans
}

func groupAnagrams1(strs []string) [][]string {
	mp := map[[26]int][]string{}
	for _, str := range strs {
		cnt := [26]int{}
		for _, b := range str {
			cnt[b-'a']++
		}
		mp[cnt] = append(mp[cnt], str)
	}
	ans := make([][]string, 0, len(mp))
	for _, v := range mp {
		ans = append(ans, v)
	}
	return ans
}

// 50
func myPow(x float64, n int) float64 {
	helper := func(x float64, n int) float64 {
		ans := float64(1)
		for n > 0 {
			if n&1 == 1 {
				ans *= x
			}
			x *= x
			n >>= 1
		}
		return ans
	}
	if n < 0 {
		return 1 / helper(x, -n)
	} else {
		return helper(x, n)
	}
}

// 51
func solveNQueens1(n int) [][]string {
	graph := make([][]bool, n)
	ans := make([][]string, 0)
	for i := range graph {
		graph[i] = make([]bool, n)
	}
	check := func(i, j int) bool {
		for k := j; k >= 0; k-- {
			if graph[i][k] {
				return false
			}
		}
		for k := i; k >= 0; k-- {
			if graph[k][j] {
				return false
			}
		}
		for k, v := i, j; k >= 0 && v >= 0; k, v = k-1, v-1 {
			if graph[k][v] {
				return false
			}
		}
		for k, v := i, j; k >= 0 && v < n; k, v = k-1, v+1 {
			if graph[k][v] {
				return false
			}
		}
		return true
	}
	var dfs func(i int)
	dfs = func(i int) {
		if i == n {
			res := make([]string, n)
			for k := 0; k < n; k++ {
				tmp := make([]byte, n)
				for v := 0; v < n; v++ {
					if graph[k][v] {
						tmp[v] = 'Q'
					} else {
						tmp[v] = '.'
					}
				}
				res[k] = string(tmp)
			}
			ans = append(ans, res)
			return
		}
		for j := 0; j < n; j++ {
			if check(i, j) {
				graph[i][j] = true
				dfs(i + 1)
				graph[i][j] = false
			}
		}
	}
	dfs(0)
	return ans
}

func solveNQueens(n int) [][]string { // 使用位图
	solutions := make([][]string, 0)
	queens := make([]int, n)
	for i := 0; i < n; i++ {
		queens[i] = -1
	}
	generateBoard := func(queens []int, n int) []string {
		board := make([]string, 0)
		for i := 0; i < n; i++ {
			row := make([]byte, n)
			for j := 0; j < n; j++ {
				row[j] = '.'
			}
			row[queens[i]] = 'Q'
			board = append(board, string(row))
		}
		return board
	}
	var solve func(queens []int, n, row, columns, diagonals1, diagonals2 int)
	solve = func(queens []int, n, row, columns, diagonals1, diagonals2 int) {
		if row == n {
			board := generateBoard(queens, n)
			solutions = append(solutions, board)
			return
		}
		availablePositions := ((1 << n) - 1) & (^(columns | diagonals1 | diagonals2))
		for availablePositions != 0 {
			position := availablePositions & (-availablePositions)
			availablePositions = availablePositions & (availablePositions - 1)
			//column := bits.OnesCount(uint(position - 1)) // 返回二进制中1的个数
			column := bits.TrailingZeros(uint(position)) // 返回最后一位1的索引（自右向左，最右为0）
			queens[row] = column
			solve(queens, n, row+1, columns|position, (diagonals1|position)>>1, (diagonals2|position)<<1)
			queens[row] = -1
		}
	}
	solve(queens, n, 0, 0, 0, 0)
	return solutions
}

// 52
func totalNQueens(n int) int {
	ans := 0
	queens := make([]int, n)
	for i := 0; i < n; i++ {
		queens[i] = -1
	}

	var solve func(queens []int, n, row, columns, diagonals1, diagonals2 int)
	solve = func(queens []int, n, row, columns, diagonals1, diagonals2 int) {
		if row == n {
			ans++
			return
		}
		availablePositions := ((1 << n) - 1) & (^(columns | diagonals1 | diagonals2))
		for availablePositions != 0 {
			position := availablePositions & (-availablePositions)
			availablePositions = availablePositions & (availablePositions - 1)
			//column := bits.OnesCount(uint(position - 1)) // 返回二进制中1的个数
			column := bits.TrailingZeros(uint(position)) // 返回最后一位1的索引（自右向左，最右为0）
			queens[row] = column
			solve(queens, n, row+1, columns|position, (diagonals1|position)>>1, (diagonals2|position)<<1)
			queens[row] = -1
		}
	}
	solve(queens, n, 0, 0, 0, 0)
	return ans
}

// 53
func maxSubArray(nums []int) int {
	sum := 0
	ans := nums[0]
	max := func(x, y int) int {
		if x > y {
			return x
		}
		return y
	}
	for _, num := range nums {
		if sum < 0 {
			sum = num
		} else {
			sum += num
		}
		ans = max(ans, sum)
	}
	return ans
}

// 54
func spiralOrder(matrix [][]int) []int {
	ans := make([]int, 0)
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return ans
	}
	m, n := len(matrix), len(matrix[0])
	up, down, left, right := 0, m-1, 0, n-1
	for true {
		for i := left; i <= right; i++ {
			ans = append(ans, matrix[up][i])
		}
		up++
		if up > down {
			break
		}
		for i := up; i <= down; i++ {
			ans = append(ans, matrix[i][right])
		}
		right--
		if right < left {
			break
		}
		for i := right; i >= left; i-- {
			ans = append(ans, matrix[down][i])
		}
		down--
		if down < up {
			break
		}
		for i := down; i >= up; i-- {
			ans = append(ans, matrix[i][left])
		}
		left++
		if left > right {
			break
		}
	}
	return ans
}

// 55
func canJump(nums []int) bool {
	end := 0
	maxDis := 0
	max := func(x, y int) int {
		if x > y {
			return x
		}
		return y
	}
	for i := 0; i < len(nums); i++ {
		maxDis = max(maxDis, i+nums[i])
		if end == i {
			end = maxDis
		} else if i > end {
			return false
		}
	}
	return true
}

// 56
func merge(intervals [][]int) [][]int {
	max := func(x, y int) int {
		if x > y {
			return x
		}
		return y
	}
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	ans := make([][]int, 0)
	for i := 0; i < len(intervals); i++ {
		if i == 0 || ans[len(ans)-1][1] < intervals[i][0] {
			ans = append(ans, intervals[i])
		} else {
			ans[len(ans)-1][1] = max(intervals[i][1], ans[len(ans)-1][1])
		}
	}
	return ans
}

// 57
func insert1(intervals [][]int, newInterval []int) [][]int {
	intervals = append(intervals, newInterval)
	return merge(intervals)
}

func insert(intervals [][]int, newInterval []int) [][]int {
	n := len(intervals)
	i := 0
	ans := make([][]int, 0)
	min := func(x, y int) int {
		if x < y {
			return x
		}
		return y
	}
	max := func(x, y int) int {
		if x > y {
			return x
		}
		return y
	}
	for ; i < n && intervals[i][1] < newInterval[0]; i++ {
		ans = append(ans, intervals[i])
	}
	for ; i < n && intervals[i][0] <= newInterval[1]; i++ {
		newInterval[0] = min(newInterval[0], intervals[i][0])
		newInterval[1] = max(newInterval[1], intervals[i][1])
	}
	ans = append(ans, newInterval)
	for ; i < n; i++ {
		ans = append(ans, intervals[i])
	}
	return ans
}

// 58
func lengthOfLastWord(s string) int {
	s = strings.Trim(s, " ") //
	//s = strings.TrimSpace(s)
	ans := strings.Split(s, " ")
	return len(ans[len(ans)-1])
}

// 59
func generateMatrix(n int) [][]int {
	ans := make([][]int, n)
	for i := range ans {
		ans[i] = make([]int, n)
	}
	left, right, up, down := 0, n-1, 0, n-1
	cnt := 1
	for cnt <= n*n {
		for i := left; i <= right; i++ {
			ans[up][i] = cnt
			cnt++
		}
		up++
		for i := up; i <= down; i++ {
			ans[i][right] = cnt
			cnt++
		}
		right--
		for i := right; i >= left; i-- {
			ans[down][i] = cnt
			cnt++
		}
		down--
		for i := down; i >= up; i-- {
			ans[i][left] = cnt
			cnt++
		}
		left++
	}
	return ans
}

// 60
func getPermutation(n int, k int) string {
	factorial := make([]int, n)
	factorial[0] = 1
	for i := 1; i < n; i++ {
		factorial[i] = factorial[i-1] * i
	}
	k--
	ans := make([]byte, n)
	rest := make([]byte, n)
	for i := 0; i < n; i++ {
		rest[i] = byte(i + '0' + 1)
	}
	for i := n; i >= 1; i-- {
		r := k % factorial[i-1]
		t := k / factorial[i-1]
		k = r
		ans[n-i] = rest[t]
		rest = append(rest[:t], rest[t+1:]...)
	}
	return string(ans)
}

// 61
func rotateRight(head *ListNode, k int) *ListNode {
	if head == nil {
		return nil
	}
	length := 0
	p := head
	for p != nil {
		length++
		if p.Next == nil {
			p.Next = head
			break
		}
		p = p.Next
	}
	k = length - k%length
	pre := new(ListNode)
	for i := 0; i < k; i++ {
		pre = head
		head = head.Next
	}
	pre.Next = nil
	return head
}

// 62
func uniquePaths(m int, n int) int {
	dp := make([]int, n)
	for i := 0; i < n; i++ {
		dp[i] = 1
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[j] += dp[j-1]
		}
	}
	return dp[n-1]
}

// 63
func uniquePathsWithObstacles1(obstacleGrid [][]int) int {
	m, n := len(obstacleGrid), len(obstacleGrid[0])
	dp := make([][]int, m)
	for i := range dp {
		dp[i] = make([]int, n)
	}
	for i := 0; i < m; i++ {
		if obstacleGrid[i][0] == 1 {
			break
		}
		dp[i][0] = 1
	}
	for j := 0; j < n; j++ {
		if obstacleGrid[0][j] == 1 {
			break
		}
		dp[0][j] = 1
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			if obstacleGrid[i][j] == 1 {
				continue
			}
			dp[i][j] = dp[i-1][j] + dp[i][j-1]
		}
	}
	return dp[m-1][n-1]
}

func uniquePathsWithObstacles(obstacleGrid [][]int) int {
	m, n := len(obstacleGrid), len(obstacleGrid[0])
	f := make([]int, n)
	if obstacleGrid[0][0] == 0 {
		f[0] = 1
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if obstacleGrid[i][j] == 1 {
				f[j] = 0
				continue
			}
			if j - 1 >= 0 && obstacleGrid[i][j-1] == 0 {
				f[j] += f[j-1]
			}
		}
	}
	return f[len(f)-1]
}

// 64
func minPathSum(grid [][]int) int {
	m := len(grid)
	n := len(grid[0])
	dp := make([][]int, m)
	for i := range dp {
		dp[i] = make([]int, n)
	}
	dp[0][0] = grid[0][0]
	for i := 1; i < m; i++ {
		dp[i][0] = grid[i][0] + dp[i-1][0]
	}
	for j := 1; j < n; j++ {
		dp[0][j] = grid[0][j] + dp[0][j-1]
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[i][j] = int(math.Min(float64(dp[i-1][j]), float64(dp[i][j-1]))) + grid[i][j]
		}
	}
	return dp[m-1][n-1]
}

// 65 有限状态机
func isNumber(s string) bool {
	helper := func(c byte) int {
		switch c {
		case ' ':
			return 0
		case '+', '-':
			return 1
		case '.':
			return 3
		case 'e', 'E':
			return 4
		default:
			if c >= 48 && c <= 57 {
				return 2
			}
		}
		return -1
	}
	state := 0
	finals := 0b101101000
	transfer := [][]int{{0, 1, 6, 2, -1},
		{-1, -1, 6, 2, -1},
		{-1, -1, 3, -1, -1},
		{8, -1, 3, -1, 4},
		{-1, 7, 5, -1, -1},
		{8, -1, 5, -1, -1},
		{8, -1, 6, 3, 4},
		{-1, -1, 5, -1, -1},
		{8, -1, -1, -1, -1}}
	for i := 0; i < len(s); i++ {
		id := helper(s[i])
		if id < 0 {
			return false
		}
		state = transfer[state][id]
		if state < 0 {
			return false
		}
	}
	return (finals & (1 << state)) > 0
}

// 66
func plusOne(digits []int) []int {
	n := len(digits)
	ans := make([]int, n+1)
	carry := 1
	for i := n; i > 0; i-- {
		sum := digits[i-1] + carry
		ans[i] = sum % 10
		carry = sum / 10
	}
	if carry == 1 {
		ans[0] = 1
		return ans
	} else {
		return ans[1:]
	}
}

func plusOne1(digits []int) []int {
	for i := len(digits) - 1; i >= 0; i-- {
		if digits[i] < 9 {
			digits[i]++
			return digits
		} else {
			digits[i] = 0
		}
	}
	return append([]int{1}, digits...)
}

// 67
func addBinary(a string, b string) string {
	m, n := len(a), len(b)
	carry := byte(0)
	sum := byte(0)
	ans := make([]byte, 0)
	for i, j := m-1, n-1; i >= 0 || j >= 0; i, j = i-1, j-1 {
		sum = carry
		if i >= 0 {
			sum += a[i] - '0'
		}
		if j >= 0 {
			sum += b[j] - '0'
		}
		ans = append([]byte{sum%2 + '0'}, ans...)
		carry = sum / 2
	}
	if carry == 1 {
		ans = append([]byte{1 + '0'}, ans...)
	}
	return string(ans)
}

// 69
func mySqrt(x int) int {
	l, r := 0, x
	for l < r {
		mid := (l+r)>>1 + 1
		if mid*mid == x {
			return mid
		} else if mid*mid < x {
			l = mid
		} else {
			r = mid - 1
		}
	}
	return l
}

// 70
func climbStairs(n int) int {
	a, b := 1, 1
	for i := 0; i < n; i++ {
		a, b = b, a+b
	}
	return a
}

// 71
func simplifyPath(path string) string {
	split := strings.Split(path, `/`)
	stack := make([]string, 0)
	for _, str := range split {
		if str == "" {
			continue
		} else if str == ".." {
			if len(stack) != 0 {
				stack = stack[:len(stack)-1]
			}
		} else if str != "." {
			stack = append(stack, str)
		}
	}
	return fmt.Sprintf("/%s", strings.Join(stack, "/"))
}

// 72
func minDistance(word1 string, word2 string) int {
	min := func(x, y int) int {
		if x < y {
			return x
		}
		return y
	}
	m := len(word1)
	n := len(word2)
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	for i := 1; i <= m; i++ {
		dp[i][0] = dp[i-1][0] + 1
	}
	for j := 1; j <= n; j++ {
		dp[0][j] = dp[0][j-1] + 1
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				dp[i][j] = min(min(dp[i-1][j], dp[i][j-1]), dp[i-1][j-1]) + 1
			}
		}
	}
	return dp[m][n]
}

// 73
func setZeroes(matrix [][]int) {
	m, n := len(matrix), len(matrix[0])
	row, col := false, false
	for i := 0; i < m; i++ {
		if matrix[i][0] == 0 {
			row = true
			break
		}
	}
	for j := 0; j < n; j++ {
		if matrix[0][j] == 0 {
			col = true
			break
		}
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			if matrix[i][j] == 0 {
				matrix[i][0] = 0
				matrix[0][j] = 0
			}
		}
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			if matrix[i][0] == 0 || matrix[0][j] == 0 {
				matrix[i][j] = 0
			}
		}
	}
	if row {
		for i := 0; i < m; i++ {
			matrix[i][0] = 0
		}
	}
	if col {
		for j := 0; j < n; j++ {
			matrix[0][j] = 0
		}
	}
}

// 74
func searchMatrix(matrix [][]int, target int) bool {
	m, n := len(matrix), len(matrix[0])
	i, j := m-1, 0
	for i >= 0 && j < n {
		if matrix[i][j] == target {
			return true
		} else if matrix[i][j] > target {
			i--
		} else if matrix[i][j] < target {
			j++
		}
	}
	return false
}

// 75
func sortColors(nums []int) {
	p0, p1 := 0, 0
	for i := 0; i < len(nums); i++ {
		if nums[i] == 0 {
			nums[i], nums[p0] = nums[p0], nums[i]
			if p0 < p1 {
				nums[i], nums[p1] = nums[p1], nums[i]
			}
			p0++
			p1++
		} else if nums[i] == 1 {
			nums[i], nums[p1] = nums[p1], nums[i]
			p1++
		}
	}
}

// 76
func minWindow1(s string, t string) string {
	mpt := make(map[byte]int, 0)
	for i := range t {
		mpt[t[i]]++
	}
	mps := make(map[byte]int, 0)
	check := func() bool {
		for k, v := range mpt {
			if mps[k] < v {
				return false
			}
		}
		return true
	}
	lp, rp := 0, 1
	ansL, ansR := 0, len(s)+1
	for ; rp <= len(s); rp++ {
		mps[s[rp-1]]++
		for ; lp <= rp && check(); lp++ {
			if rp-lp < ansR-ansL {
				ansR, ansL = rp, lp
			}
			mps[s[lp]]--
		}
	}
	if ansR == len(s)+1 {
		return ""
	}
	return s[ansL:ansR]
}

func minWindow(s string, t string) string {
	need, have := make([]int, 128), make([]int, 128)
	for i := range t {
		need[t[i]]++
	}
	sLen, tLen := len(s), len(t)
	l, r := 0, 0
	minLen, start, count := sLen+1, 0, 0
	for ; r < sLen; r++ {
		if have[s[r]] < need[s[r]] {
			count++
		}
		have[s[r]]++
		for ; count == tLen; l++ {
			if r-l < minLen {
				minLen = r - l + 1
				start = l
			}
			if have[s[l]] == need[s[l]] {
				count--
			}
			have[s[l]]--
		}
	}
	if minLen == sLen+1 {
		return ""
	}
	return s[start : start+minLen]
}

// 77
func combine(n int, k int) [][]int {
	ans := make([][]int, 0)
	path := make([]int, 0)
	var dfs func(cur int)
	dfs = func(cur int) {
		if len(path) == k {
			tmp := make([]int, k)
			copy(tmp, path)
			ans = append(ans, tmp)
			return
		}
		for i := cur; i <= n; i++ {
			path = append(path, i)
			dfs(i + 1)
			path = path[:len(path)-1]
		}
	}
	dfs(1)
	return ans
}

// 78
func subsets(nums []int) [][]int {
	ans := make([][]int, 0)
	path := make([]int, 0)
	var dfs func(cur int)
	dfs = func(cur int) {
		tmp := make([]int, len(path))
		copy(tmp, path)
		ans = append(ans, tmp)
		for i := cur; i < len(nums); i++ {
			path = append(path, nums[i])
			dfs(i + 1)
			path = path[:len(path)-1]
		}
	}
	dfs(0)
	return ans
}

// wjq: dfs返回值问题
// 只需找到一个结果时，可设置dfs返回值为bool，当为true时结束搜索
// 79 剑指offer12
func exist1(board [][]byte, word string) bool {
	flag := false
	m, n := len(board), len(board[0])
	used := make([][]bool, m)
	for i := range used {
		used[i] = make([]bool, n)
	}
	dx := []int{0, 1, 0, -1}
	dy := []int{1, 0, -1, 0}
	var dfs func(i, j int, cur int)
	dfs = func(i, j int, cur int) {
		if cur == len(word) {
			flag = true
			return
		}
		if i < 0 || i >= m || j < 0 || j >= n || board[i][j] != word[cur] || flag || used[i][j] {
			return
		}
		used[i][j] = true
		for k := 0; k < 4; k++ {
			nx := i + dx[k]
			ny := j + dy[k]
			dfs(nx, ny, cur+1)
		}
		used[i][j] = false
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if !flag {
				dfs(i, j, 0)
			}
		}
	}
	return flag
}

func exist(board [][]byte, word string) bool {
	// 先做一遍遍历，过滤word中的字符不在map中的情况
	m, n := len(board), len(board[0])
	temp := make(map[byte]bool, 0)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			temp[board[i][j]] = true
		}
	}
	for i := 0; i < len(word); i += 1 {
		_, ok := temp[word[i]]
		if !ok {
			return false
		}
	}
	used := make([][]bool, m)
	for i := range used {
		used[i] = make([]bool, n)
	}
	dx := []int{0, 1, 0, -1}
	dy := []int{1, 0, -1, 0}
	var dfs func(i, j int, cur int) bool
	dfs = func(i, j int, cur int) bool {
		if cur == len(word) {
			return true
		}
		if i < 0 || i >= m || j < 0 || j >= n || board[i][j] != word[cur] || used[i][j] {
			return false
		}
		used[i][j] = true
		for k := 0; k < 4; k++ {
			nx := i + dx[k]
			ny := j + dy[k]
			if dfs(nx, ny, cur+1) {
				return true
			}
		}
		used[i][j] = false
		return false
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if dfs(i, j, 0) {
				return true
			}
		}
	}
	return false
}

// 80
func removeDuplicates80(nums []int) int {
	n := len(nums)
	if n <= 2 {
		return n
	}
	slow, fast := 2, 2
	for ; fast < n; fast++ {
		if nums[fast] != nums[slow-2] {
			nums[slow] = nums[fast]
			slow++
		}
	}
	return slow
}

// 81
func search81(nums []int, target int) bool {
	l, r := 0, len(nums)
	for l < r {
		mid := (l + r) >> 1
		if nums[mid] == target {
			return true
		}
		if nums[l] == nums[mid] {
			l++
		} else if nums[l] < nums[mid] {
			if nums[l] <= target && target < nums[mid] {
				r = mid
			} else {
				l = mid + 1
			}
		} else if nums[l] > nums[mid] {
			if nums[mid] < target && target <= nums[r-1] {
				l = mid + 1
			} else {
				r = mid
			}
		}
	}
	return false
}

// 82
func deleteDuplicates(head *ListNode) *ListNode {
	dummy := new(ListNode)
	dummy.Next = head
	cur := dummy
	for cur.Next != nil && cur.Next.Next != nil {
		if cur.Next.Val == cur.Next.Next.Val {
			x := cur.Next.Val
			for cur.Next != nil && cur.Next.Val == x {
				cur.Next = cur.Next.Next
			}
		} else {
			cur = cur.Next
		}
	}
	return dummy.Next
}

// 83
func deleteDuplicates83(head *ListNode) *ListNode {
	cur := head
	for cur != nil && cur.Next != nil {
		if cur.Val == cur.Next.Val {
			cur.Next = cur.Next.Next
		} else {
			cur = cur.Next
		}
	}
	return head
}

// 84 忘记抄的谁的了，别人都是heights都是前后各加一个0，他只在最后加了个0，然后stack中加了个-1的索引
// 很机智。。。
// 额。。。好像是我写的？
func largestRectangleArea(heights []int) int {
	max := func(x, y int) int {
		if x > y {
			return x
		}
		return y
	}
	heights = append(heights, 0)
	n := len(heights)
	stack := make([]int, 0)
	stack = append(stack, -1)
	ans := 0
	for i := 0; i < n; i++ {
		for len(stack) > 1 && heights[i] < heights[stack[len(stack)-1]] {
			curHeight := heights[stack[len(stack)-1]]
			stack = stack[:len(stack)-1]
			curWidth := i - stack[len(stack)-1] - 1
			ans = max(ans, curWidth*curHeight)
		}
		stack = append(stack, i)
	}
	return ans
}

func largestRectangleArea1(heights []int) int {
	max := func(x, y int) int {
		if x > y {
			return x
		}
		return y
	}
	heights = append(heights, 0)
	heights = append([]int{0}, heights...)
	n := len(heights)
	stack := make([]int, 0)
	ans := 0
	for i := 0; i < n; i++ {
		for len(stack) > 1 && heights[i] < heights[stack[len(stack)-1]] {
			curHeight := heights[stack[len(stack)-1]]
			stack = stack[:len(stack)-1]
			curWidth := i - stack[len(stack)-1] - 1
			ans = max(ans, curWidth*curHeight)
		}
		stack = append(stack, i)
	}
	return ans
}

// 85
func maximalRectangle(matrix [][]byte) int {
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return 0
	}
	ans := 0
	m, n := len(matrix), len(matrix[0])
	cache := make([]int, n)
	helper := func(cur []byte) []int {
		res := make([]int, len(cache))
		for i := 0; i < len(cache); i++ {
			if cur[i] == '0' {
				res[i] = 0
			} else {
				res[i] = cache[i] + int(cur[i]-'0')
			}
		}
		return res
	}
	for i := 0; i < m; i++ {
		tmp := matrix[i]
		res := helper(tmp)
		cache = res
		ans = max(ans, largestRectangleArea(res))
	}
	return ans
}

// 86
func partition(head *ListNode, x int) *ListNode {
	l1 := new(ListNode)
	l2 := new(ListNode)
	p, q := l1, l2
	for ; head != nil; head = head.Next {
		if head.Val < x {
			p.Next = head
			p = p.Next
		} else {
			q.Next = head
			q = q.Next
		}
	}
	q.Next = nil
	p.Next = l2.Next
	return l1.Next
}

// 87
func isScramble(s1 string, s2 string) bool {
	n := len(s1)
	dp := make([][][]int8, n)
	for i := range dp {
		dp[i] = make([][]int8, n)
		for j := range dp[i] {
			dp[i][j] = make([]int8, n+1)
			for k := range dp[i][j] {
				dp[i][j][k] = -1
			}
		}
	}

	var dfs func(i1, i2, length int) int8
	dfs = func(i1, i2, length int) (res int8) {
		d := &dp[i1][i2][length]
		if *d != -1 {
			return *d
		}
		defer func() {
			*d = res
		}()
		x, y := s1[i1:i1+length], s2[i2:i2+length]
		if x == y {
			return 1
		}
		freq := make([]int, 26)
		for i := 0; i < len(x); i++ {
			freq[x[i]-'a']++
			freq[y[i]-'a']--
		}
		for _, f := range freq {
			if f != 0 {
				return 0
			}
		}
		for i := 1; i < length; i++ {
			if dfs(i1, i2, i) == 1 && dfs(i1+i, i2+i, length-i) == 1 {
				return 1
			}
			if dfs(i1, i2+length-i, i) == 1 && dfs(i1+i, i2, length-i) == 1 {
				return 1
			}
		}
		return 0
	}
	return dfs(0, 0, n) == 1
}

// 88
func merge88(nums1 []int, m int, nums2 []int, n int) {
	for i, j, k := m-1, n-1, m+n-1; i >= 0 || j >= 0; k-- {
		if i == -1 {
			nums1[k] = nums2[j]
			j--
		} else if j == -1 {
			nums1[k] = nums1[i]
			i--
		} else if nums1[i] > nums2[j] {
			nums1[k] = nums1[i]
			i--
		} else {
			nums1[k] = nums2[j]
			j--
		}
	}
}

// 89
func grayCode(n int) []int {
	ans := make([]int, 1)
	head := 1
	for i := 0; i < n; i++ {
		for j := len(ans) - 1; j >= 0; j-- {
			ans = append(ans, head|ans[j])
		}
		head <<= 1
	}
	return ans
}

// 90
func subsetsWithDup(nums []int) [][]int {
	sort.Ints(nums)
	ans := make([][]int, 0)
	path := make([]int, 0)
	var dfs func(cur int)
	dfs = func(cur int) {
		tmp := make([]int, len(path))
		copy(tmp, path)
		ans = append(ans, tmp)
		for i := cur; i < len(nums); i++ {
			if i != cur && nums[i] == nums[i-1] {
				continue
			}
			path = append(path, nums[i])
			dfs(i + 1)
			path = path[:len(path)-1]
		}
	}
	dfs(0)
	return ans
}

// 91
func numDecodings(s string) int {
	if len(s) == 0 || s[0] == '0' {
		return 0
	}
	a, b := 1, 1
	for i := 1; i < len(s); i++ {
		cur := 0
		if s[i-1] == '1' || (s[i-1] == '2' && s[i] <= '6') {
			cur += a
		}
		if s[i] != '0' {
			cur += b
		}
		a, b = b, cur
	}
	return b
}

func numDecodings1(s string) int {
	n := len(s)
	// a = f[i-2], b = f[i-1], c = f[i]
	a, b, c := 0, 1, 0
	for i := 0; i < n; i++ {
		c = 0
		if s[i] != '0' {
			c += b
		}
		if i > 0 && (s[i-1] == '1' || (s[i-1] == '2' && s[i] <= '6')) {
			c += a
		}
		a, b = b, c
	}
	return c
}

// 92
func reverseBetween(head *ListNode, left int, right int) *ListNode {
	helper := func(head *ListNode) *ListNode {
		pre := new(ListNode)
		cur := head
		for cur != nil {
			next := cur.Next
			cur.Next = pre
			pre = cur
			cur = next
		}
		return pre
	}
	dummy := new(ListNode)
	dummy.Next = head
	preNode := dummy
	for i := 0; i < left-1; i++ {
		preNode = preNode.Next
	}
	startNode := preNode.Next
	endNode := startNode
	for i := 0; i < right-left; i++ {
		endNode = endNode.Next
	}
	afterNode := endNode.Next
	preNode.Next = nil
	endNode.Next = nil
	preNode.Next = helper(startNode)
	startNode.Next = afterNode
	return dummy.Next
}

// 93
func restoreIpAddresses(s string) []string {
	n := len(s)
	ans := make([]string, 0)
	//if n < 4 || n > 12 {
	//	return ans
	//}
	check := func(l, r int) bool {
		str := s[l:r]
		if len(str) > 1 && str[0] == '0' {
			return false
		}
		v, _ := strconv.Atoi(str)
		return v >= 0 && v <= 255
	}
	path := make([]string, 0)
	var dfs func(int, int)
	dfs = func(cnt, cur int) {
		if cur == n {
			if cnt == 4 {
				ans = append(ans, strings.Join(path, "."))
			}
			return
		}
		if n-cur < 4-cnt || n-cur > 3*(4-cnt) {
			return
		}
		for i := 1; i <= 3; i++ {
			if cur+i > n {
				break
			}
			if check(cur, cur+i) {
				str := s[cur : cur+i]
				path = append(path, str)
				dfs(cnt+1, cur+i)
				path = path[:len(path)-1]
			}
		}
	}
	dfs(0, 0)
	return ans
}

// 94
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func inorderTraversal(root *TreeNode) []int {
	ans := make([]int, 0)
	var helper func(node *TreeNode)
	helper = func(root *TreeNode) {
		if root == nil {
			return
		}
		helper(root.Left)
		ans = append(ans, root.Val)
		helper(root.Right)
	}
	helper(root)
	return ans
}

// 95
func generateTrees(n int) []*TreeNode {
	var helper func(start, end int) []*TreeNode
	helper = func(start, end int) []*TreeNode {
		ans := make([]*TreeNode, 0)
		if start > end {
			ans = append(ans, nil)
			return ans
		}
		for i := start; i <= end; i++ {
			left := helper(start, i-1)
			right := helper(i+1, end)
			for _, l := range left {
				for _, r := range right {
					cur := &TreeNode{
						Val:   i,
						Left:  l,
						Right: r,
					}
					ans = append(ans, cur)
				}
			}
		}
		return ans
	}
	return helper(1, n)
}

// 96
func numTrees(n int) int { // 卡塔兰数
	C := 1
	for i := 0; i < n; i++ {
		C = C * 2 * (2*i + 1) / (i + 2)
	}
	return C
}

// 97
func isInterleave(s1 string, s2 string, s3 string) bool {
	m, n, t := len(s1), len(s2), len(s3)
	if (m + n) != t {
		return false
	}
	f := make([][]bool, m+1)
	for i := 0; i <= m; i++ {
		f[i] = make([]bool, n+1)
	}
	f[0][0] = true
	for i := 0; i <= m; i++ {
		for j := 0; j <= n; j++ {
			p := i + j - 1
			if i > 0 {
				f[i][j] = f[i][j] || (f[i-1][j] && s1[i-1] == s3[p])
			}
			if j > 0 {
				f[i][j] = f[i][j] || (f[i][j-1] && s2[j-1] == s3[p])
			}
		}
	}
	return f[m][n]
}

// 98
func isValidBST(root *TreeNode) bool {
	var helper func(*TreeNode, int, int) bool
	helper = func(root *TreeNode, l, r int) bool {
		if root == nil {
			return true
		}
		if root.Val <= l || root.Val >= r {
			return false
		}
		return helper(root.Left, l, root.Val) && helper(root.Right, root.Val, r)
	}
	return helper(root, math.MinInt64, math.MaxInt64)
}

// 99
func recoverTree(root *TreeNode) {
	var x, y, parent *TreeNode
	var helper func(*TreeNode)
	helper = func(root *TreeNode) {
		if root == nil {
			return
		}
		helper(root.Left)
		if parent != nil && parent.Val > root.Val {
			y = root
			if x == nil {
				x = parent
			} else {
				return
			}
		}
		parent = root
		helper(root.Right)
	}
	helper(root)
	x.Val, y.Val = y.Val, x.Val
}

// 100
func isSameTree(p *TreeNode, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	}
	if p == nil || q == nil {
		return false
	}
	if p.Val != q.Val {
		return false
	}
	return isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
}
//func main() {
//	t1 := &TreeNode{
//		Val:   1,
//		Left:  nil,
//		Right: nil,
//	}
//	t2 := &TreeNode{
//		Val:   2,
//		Left:  nil,
//		Right: nil,
//	}
//	t3 := &TreeNode{
//		Val:   3,
//		Left:  nil,
//		Right: nil,
//	}
//	t1.Left = t3
//	t3.Right = t2
//	recoverTree(t1)
//}
