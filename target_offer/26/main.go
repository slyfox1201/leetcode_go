package main

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func isSubStructure(A *TreeNode, B *TreeNode) bool {
	return (A != nil && B != nil) && (helper(A, B) || isSubStructure(A.Left, B) || isSubStructure(A.Right, B))
}

func helper(A *TreeNode, B *TreeNode) bool {
	if B == nil {
		return true
	}
	if A == nil || A.Val != B.Val {
		return false
	}
	return helper(A.Left, B.Left) && helper(A.Right, B.Right)
}
