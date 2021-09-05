package main

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// 101
func isSymmetric(root *TreeNode) bool {
	var helper func(left, right *TreeNode) bool
	helper = func(left, right *TreeNode) bool {
		if left == nil && right == nil {
			return true
		}
		if left == nil || right == nil {
			return false
		}
		if left.Val != right.Val {
			return false
		}
		return helper(left.Left, right.Right) && helper(left.Right, right.Left)
	}
	return helper(root, root)
}

// 102
func levelOrder(root *TreeNode) [][]int {
	ans := make([][]int, 0)
	queue := make([]*TreeNode, 0)
	if root != nil {
		queue = append(queue, root)
	}
	for len(queue) != 0 {
		level := make([]int, 0)
		size := len(queue)
		for i := 0; i < size; i++ {
			node := queue[0]
			queue = queue[1:]
			level = append(level, node.Val)
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		ans = append(ans, level)
	}
	return ans
}

// 103
func zigzagLevelOrder(root *TreeNode) [][]int {
	reverse := func(arr []int) {
		for i, j := 0, len(arr)-1; i < j; i, j = i+1, j-1 {
			arr[i], arr[j] = arr[j], arr[i]
		}
	}
	ans := make([][]int, 0)
	queue := make([]*TreeNode, 0)
	if root != nil {
		queue = append(queue, root)
	}
	for len(queue) != 0 {
		level := make([]int, 0)
		size := len(queue)
		for i := 0; i < size; i++ {
			node := queue[0]
			queue = queue[1:]
			level = append(level, node.Val)
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		ans = append(ans, level)
	}
	for i, row := range ans {
		if i&1 == 1 {
			reverse(row)
		}
	}
	return ans
}

// 104
func maxDepth(root *TreeNode) int {
	max := func(x, y int) int {
		if x > y {
			return x
		}
		return y
	}
	var helper func(*TreeNode) int
	helper = func(root *TreeNode) int {
		if root == nil {
			return 0
		}
		return max(helper(root.Left), helper(root.Right)) + 1
	}
	return helper(root)
}

// 105
func buildTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 {
		return nil
	}
	root := &TreeNode{Val: preorder[0]}
	i := 0
	for ; i < len(preorder); i++ {
		if preorder[0] == inorder[i] {
			break
		}
	}
	root.Left = buildTree(preorder[1:i+1], inorder[:i])
	root.Right = buildTree(preorder[i+1:], inorder[i+1:])
	return root
}

// 106
func buildTree106(inorder []int, postorder []int) *TreeNode {
	if len(postorder) == 0 {
		return nil
	}
	root := &TreeNode{Val: postorder[len(postorder)-1]}
	i := 0
	for ; i < len(postorder); i++ {
		if postorder[len(postorder)-1] == inorder[i] {
			break
		}
	}
	root.Left = buildTree106(inorder[:i], postorder[:i])
	root.Right = buildTree106(inorder[i+1:], postorder[i:len(postorder)-1])
	return root
}

// 107
func levelOrderBottom(root *TreeNode) [][]int {
	reverse := func(arr [][]int) {
		for i, j := 0, len(arr)-1; i < j; i, j = i+1, j-1 {
			arr[i], arr[j] = arr[j], arr[i]
		}
	}
	ans := make([][]int, 0)
	queue := make([]*TreeNode, 0)
	if root != nil {
		queue = append(queue, root)
	}
	for len(queue) != 0 {
		level := make([]int, 0)
		size := len(queue)
		for i := 0; i < size; i++ {
			node := queue[0]
			queue = queue[1:]
			level = append(level, node.Val)
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		ans = append(ans, level)
	}
	reverse(ans)
	return ans
}

// 108
func sortedArrayToBST(nums []int) *TreeNode {
	var helper func(int, int) *TreeNode
	helper = func(l, r int) *TreeNode {
		if l > r {
			return nil
		}
		mid := (l + r) / 2
		root := &TreeNode{Val: nums[mid]}
		root.Left = helper(l, mid-1)
		root.Right = helper(mid+1, r)
		return root
	}
	return helper(0, len(nums)-1)
}
