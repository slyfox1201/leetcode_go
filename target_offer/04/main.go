package main

func findNumberIn2DArray(matrix [][]int, target int) bool {
	rows := len(matrix)
	if rows == 0 {
		return false
	}
	cols := len(matrix[0])
	for i, j := rows-1, 0; i >= 0 && j < cols; {
		if matrix[i][j] == target {
			return true
		} else if matrix[i][j] < target {
			j++
		} else {
			i--
		}
	}
	return false
}
