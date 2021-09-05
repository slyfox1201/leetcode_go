package main

func verifyPostorder(postorder []int) bool {
	return helper(postorder, 0, len(postorder)-1)
}

func helper(postorder []int, left, right int) bool {
	if left >= right {
		return true
	}
	mid := left
	for postorder[mid] < postorder[right] {
		mid++
	}
	for tmp := mid; tmp < right; tmp++ {
		if postorder[tmp] < postorder[right] {
			return false
		}
	}
	return helper(postorder, left, mid-1) && helper(postorder, mid, right-1)
}
