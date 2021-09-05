package main

import (
	"bytes"
	"container/list"
	"fmt"
	"strconv"
	"strings"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func serialize(root *TreeNode) string {
	if root == nil {
		return "[]"
	}
	res := new(bytes.Buffer)
	res.WriteByte('[')
	queue := list.List{}
	queue.PushBack(root)
	for queue.Len() != 0 {
		node := queue.Remove(queue.Front()).(*TreeNode)
		if node != nil {
			res.WriteString(strconv.Itoa(node.Val) + ",")
			queue.PushBack(node.Left)
			queue.PushBack(node.Right)
		} else {
			res.WriteString("null,")
		}
	}
	res.Truncate(res.Len() - 1)
	res.WriteByte(']')
	return res.String()
}

func deserialize(data string) *TreeNode {
	if data == "[]" {
		return nil
	}
	vals := strings.Split(data[1:len(data)-1], ",")
	val, _ := strconv.Atoi(vals[0])
	root := &TreeNode{Val: val}
	queue := list.List{}
	queue.PushBack(root)
	i := 1
	for queue.Len() != 0 {
		node := queue.Remove(queue.Front()).(*TreeNode)
		if vals[i] != "null" {
			val, _ := strconv.Atoi(vals[i])
			node.Left = &TreeNode{Val: val}
			queue.PushBack(node.Left)
		}
		i++
		if vals[i] != "null" {
			val, _ := strconv.Atoi(vals[i])
			node.Right = &TreeNode{Val: val}
			queue.PushBack(node.Right)
		}
		i++
	}
	return root
}

func main() {
	node1 := &TreeNode{Val: 1}
	node2 := &TreeNode{Val: 2}
	node3 := &TreeNode{Val: 3}
	node4 := &TreeNode{Val: 4}
	node5 := &TreeNode{Val: 5}
	node1.Left = node2
	node1.Right = node3
	node3.Left = node4
	node3.Right = node5
	res := serialize(node1)
	fmt.Println(res)
	node := deserialize(res)
	fmt.Println(node)
}
