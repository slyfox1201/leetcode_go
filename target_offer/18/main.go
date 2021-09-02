package main

type ListNode struct {
	Val  int
	Next *ListNode
}

func deleteNode(head *ListNode, val int) *ListNode {
	dummy := &ListNode{}
	dummy.Next = head
	pre, cur := dummy, head
	for cur !=nil {
		if cur.Val == val {
			pre.Next = cur.Next
			break
		} else {
			cur = cur.Next
			pre = pre.Next
		}
	}
	return dummy.Next
}
