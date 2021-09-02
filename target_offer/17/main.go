package main

import "math"

func printNumbers(n int) []int {
	end := int(math.Pow(10, float64(n))) - 1
	res := make([]int, end)
	for i := range res {
		res[i] = i + 1
	}
	return res
}
