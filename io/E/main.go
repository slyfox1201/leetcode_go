package main

import "fmt"

func main() {
	var n, m int
	fmt.Scan(&n)
	for i := 0; i < n; i++ {
		fmt.Scan(&m)
		sum, tmp := 0, 0
		for j := 0; j < m; j++ {
			fmt.Scan(&tmp)
			sum += tmp
		}
		fmt.Println(sum)
	}
}
