package main

type ListNode struct {
	Val  int
	Next *ListNode
}

func reversePrint1(head *ListNode) []int {
	res := make([]int, 0)
	for ; head != nil; head = head.Next {
		res = append([]int{head.Val}, res...)
	}
	return res
}

func reversePrint(head *ListNode) []int {
	res := make([]int, 0)
	for ; head != nil; head = head.Next {
		res = append(res, head.Val)
	}
	l := len(res)
	for i := 0; i < l/2; i++ {
		swap(res, i, l-i-1)
	}
	return res
}

func swap(arr []int, x, y int) {
	arr[x], arr[y] = arr[y], arr[x]
}
