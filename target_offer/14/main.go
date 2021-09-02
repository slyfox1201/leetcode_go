package main

// dp
func cuttingRope(n int) int {
	dp := make([]int, n+1)
	dp[2] = 1
	for i := 3; i < n+1; i++ {
		for j := 2; j < i; j++ {
			dp[i] = max(dp[i], max(j*(i-j), j*dp[i-j]))
		}
	}
	return dp[n]
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

// 贪心
func cuttingRope1(n int) int {
	if n < 4 {
		return n - 1
	}
	res := 1
	for n > 4 {
		res *= 3
		n -= 3
	}
	return res * n
}
