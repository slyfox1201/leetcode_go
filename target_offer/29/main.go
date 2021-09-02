package main

func spiralOrder(matrix [][]int) []int {
	m := len(matrix)
	if m == 0 {
		return nil
	}
	n := len(matrix[0])
	res := make([]int, m*n)
	i := 0
	top, right, bottom, left := 0, n-1, m-1, 0
	for {
		for k := left; k <= right; k++ {
			res[i] = matrix[top][k]
			i++
		}
		top++
		if top > bottom {
			break
		}
		for k := top; k <= bottom; k++ {
			res[i] = matrix[k][right]
			i++
		}
		right--
		if left > right {
			break
		}
		for k := right; k >= left; k-- {
			res[i] = matrix[bottom][k]
			i++
		}
		bottom--
		if top > bottom {
			break
		}
		for k := bottom; k >= top; k-- {
			res[i] = matrix[k][left]
			i++
		}
		left++
		if left > right {
			break
		}
	}
	return res
}
