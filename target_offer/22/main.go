package main

type ListNode struct {
	Val  int
	Next *ListNode
}

func getKthFromEnd(head *ListNode, k int) *ListNode {
	s, f := head, head
	for i := 0; i < k; i++ {
		f = f.Next
	}
	for f != nil {
		f = f.Next
		s = s.Next
	}
	return s
}
