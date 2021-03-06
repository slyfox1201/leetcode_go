package main

// 关键点：不含重复数字

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func buildTree1(preorder []int, inorder []int) *TreeNode {
	if preorder == nil || len(preorder) == 0 {
		return nil
	}

	cache := make(map[int]int)
	length := len(inorder)
	for i := 0; i < length; i++ {
		cache[inorder[i]] = i
	}
	root := helper(preorder, 0, length-1, inorder, 0, length-1, cache)
	return root
}
// inEnd没有用。这个方法存在冗余
func helper(preorder []int, preStart int, preEnd int, inorder []int, inStart int, inEnd int, cache map[int]int) *TreeNode {
	if preStart > preEnd {
		return nil
	}

	rootVal := preorder[preStart]
	root := &TreeNode{Val: rootVal}
	if preStart == preEnd {
		return root
	} else {
		rootIndex := cache[rootVal]
		leftNodes := rootIndex - inStart
		left := helper(preorder, preStart+1, preStart+leftNodes, inorder, inStart, inStart+leftNodes-1, cache)
		right := helper(preorder, preStart+leftNodes+1, preEnd, inorder, rootIndex+1, inEnd, cache)
		root.Left = left
		root.Right = right
		return root
	}
}

func buildTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 {
		return nil
	}
	root := &TreeNode{preorder[0], nil, nil}
	i := 0
	// 可用map优化，懒得改了
	for ; i < len(inorder); i++ {
		if inorder[i] == preorder[0] {
			break
		}
	}
	root.Left = buildTree(preorder[1:i+1], inorder[:i])
	root.Right = buildTree(preorder[i+1:], inorder[i+1:])
	return root
}
