package main
// 与13题的区别
// 12在注1处将board恢复原样，是因为后续路径在dfs时，可能用到当前路径所占用的节点。即，board中的每个节点会被多次访问
// 13不需要恢复，是因为题目为岛屿染色，即，要求每个节点只被访问一次
var dx = []int{0, 1, 0, -1}
var dy = []int{1, 0, -1, 0}

func exist(board [][]byte, word string) bool {
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[0]); j++ {
			if dfs(board, word, i, j, 0) {
				return true
			}
		}
	}
	return false
}

// 非染色，且有一条满足条件即可完成要求，则可以返回bool，写成如下形式
func dfs(board [][]byte, word string, i, j, k int) bool {
	if i < 0 || i >= len(board) || j < 0 || j >= len(board[0]) || board[i][j] != word[k] {
		return false
	}
	if k == len(word)-1 {
		return true
	}
	board[i][j] = ' '
	for t := 0; t < 4; t++ {
		nx := i + dx[t]
		ny := j + dy[t]
		if dfs(board, word, nx, ny, k+1) {
			return true
		}
	}
	// 注1
	board[i][j] = word[k]
	return false
}
