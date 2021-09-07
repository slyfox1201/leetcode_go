package main

import "fmt"

func main() {
	ans := make([]int, 0)
	fmt.Printf("%p", &ans)
	var a []int
	fmt.Printf("%p", &a)
}
