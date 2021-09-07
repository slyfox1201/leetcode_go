package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func main1() {
	input := bufio.NewScanner(os.Stdin)
	for input.Scan() {
		splits := strings.Split(input.Text(), " ")
		if splits[0] == "0" {
			break
		}
		sum := 0
		for _, s := range splits {
			tmp, _ := strconv.Atoi(s)
			sum += tmp
		}
		fmt.Println(sum)
	}
}

func main2() {
	var n int
	for {
		fmt.Scan(&n)
		if n == 0 {
			break
		}
		sum := 0
		var tmp int
		for i := 0; i < n; i++ {
			fmt.Scan(&tmp)
			sum += tmp
		}
		fmt.Println(sum)
	}
}
