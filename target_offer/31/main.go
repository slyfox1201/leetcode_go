package main

func validateStackSequences(pushed []int, popped []int) bool {
	n := len(pushed)
	stack := make([]int, 0, n)
	j := 0
	for i := 0; i < n; i++ {
		stack = append(stack, pushed[i])
		for len(stack) > 0 && stack[len(stack)-1] == popped[j] {
			stack = stack[:len(stack)-1]
			j++
		}
	}
	return len(stack) == 0
}
