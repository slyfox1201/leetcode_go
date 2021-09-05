package main

type Node struct {
	Val   int
	Left  *Node
	Right *Node
}

var pre, head *Node

func treeToDoublyList(root *Node) *Node {
	if root == nil {
		return nil
	}
	dfs(root)
	head.Left = pre
	pre.Right = head
	return head
}

func dfs(cur *Node) {
	if cur == nil {
		return
	}
	dfs(cur.Left)
	if pre != nil {
		pre.Right = cur
	} else {
		head = cur
	}
	cur.Left = pre
	pre = cur
	dfs(cur.Right)
}
