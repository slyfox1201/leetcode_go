package main

var dx = []int{0, 1, 0, -1}
var dy = []int{1, 0, -1, 0}
var ans = 0
var visited [][]bool

func movingCount1(m int, n int, k int) int {
	// golang特色，记得置零，不然多个用例时会累计... 所以全局变量还是少用好啊
	ans = 0
	visited = make([][]bool, m)
	for i := range visited {
		visited[i] = make([]bool, n)
	}
	dfs1(m, n, 0, 0, k)
	return ans
}

// 染色不需要有返回值，只需埋头dfs
func dfs1(m, n, i, j, k int) {
	if i < 0 || i >= m || j < 0 || j >= n || visited[i][j] || add(i)+add(j) > k {
		return
	}
	ans++
	visited[i][j] = true
	// 本题只需两个方向dfs即可
	for t := 0; t < 2; t++ {
		nx := i + dx[t]
		ny := j + dy[t]
		dfs1(m, n, nx, ny, k)
	}
}

func add(num int) int {
	sum := 0
	for num != 0 {
		sum += num % 10
		num /= 10
	}
	return sum
}

// 这个版本没用全局变量，更推荐哦~
func movingCount(m int, n int, k int) int {
	dp := make([][]bool, m)
	for i := range dp {
		dp[i] = make([]bool, n)
	}
	return dfs(m, n, 0, 0, k, dp)
}

func dfs(m, n, i, j, k int, dp [][]bool) int {
	if i < 0 || j < 0 || i >= m || j >= n || dp[i][j] || (add(i)+add(j)) > k {
		return 0
	}
	dp[i][j] = true
	sum := 1
	sum += dfs(m, n, i, j+1, k, dp)
	//sum += dfs(m, n, i, j-1, k, dp)
	sum += dfs(m, n, i+1, j, k, dp)
	//sum += dfs(m, n, i-1, j, k, dp)
	return sum
}
