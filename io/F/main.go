package main

import "fmt"

func main() {
	var m int
	for {
		_, err := fmt.Scan(&m)
		if err != nil {
			break
		}
		sum, tmp := 0, 0
		for j := 0; j < m; j++ {
			fmt.Scan(&tmp)
			sum += tmp
		}
		fmt.Println(sum)
	}
}
