package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func main1() {
	var a, b int
	for {
		fmt.Scan(&a, &b)
		if a == 0 && b == 0 {
			break
		}
		fmt.Println(a + b)
	}
}

func main2() {
	input := bufio.NewScanner(os.Stdin)
	for input.Scan() {
		splits := strings.Split(input.Text(), " ")
		if splits[0] == "0" && splits[1] == "0" {
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
