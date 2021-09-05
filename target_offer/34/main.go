package main

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func pathSum1(root *TreeNode, target int) [][]int {
	res := make([][]int, 0)
	dfs(&res, root, target, 0, []int{})
	return res
}

func dfs(res *[][]int, root *TreeNode, target int, sum int, path []int) {
	if root == nil {
		return
	}
	sum += root.Val
	path = append(path, root.Val)

	if root.Left == nil && root.Right == nil {
		if sum == target {
			tmp := make([]int, len(path))
			copy(tmp, path)
			*res = append(*res, tmp)
		}
		return
	}
	dfs(res, root.Left, target, sum, path)
	dfs(res, root.Right, target, sum, path)
	path = path[:len(path)-1]
}

func pathSum(root *TreeNode, target int) (ans [][]int) {
	path := []int{}
	var dfs func(*TreeNode, int)
	dfs = func(node *TreeNode, left int) {
		if node == nil {
			return
		}
		left -= node.Val
		path = append(path, node.Val)
		defer func() { path = path[:len(path)-1] }()
		if node.Left == nil && node.Right == nil && left == 0 {
			ans = append(ans, append([]int(nil), path...))
			return
		}
		dfs(node.Left, left)
		dfs(node.Right, left)
	}
	dfs(root, target)
	return
}
