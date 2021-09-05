package main

type Node struct {
	Val    int
	Next   *Node
	Random *Node
}

func copyRandomList(head *Node) *Node {
	cache := make(map[*Node]*Node)
	var helper func(*Node) *Node
	helper = func(node *Node) *Node {
		if node == nil {
			return nil
		}
		if n, ok := cache[node]; ok {
			return n
		}
		newNode := &Node{Val: node.Val}
		cache[node] = newNode
		newNode.Next = helper(node.Next)
		newNode.Random = helper(node.Random)
		return newNode
	}
	return helper(head)
}

// 很难想到，看看就好
func copyRandomList1(head *Node) *Node {
	if head == nil {
		return nil
	}
	for node := head; node != nil; node = node.Next.Next {
		node.Next = &Node{Val: node.Val, Next: node.Next}
	}
	for node := head; node != nil; node = node.Next.Next {
		if node.Random != nil {
			node.Next.Random = node.Random.Next
		}
	}
	headNew := head.Next
	for node := head; node != nil; node = node.Next {
		nodeNew := node.Next
		node.Next = node.Next.Next
		if nodeNew.Next != nil {
			nodeNew.Next = nodeNew.Next.Next
		}
	}
	return headNew
}
