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
	input.Scan()
	for input.Scan() {
		splits := strings.Split(input.Text(), " ")
		sum := 0
		for _, s := range splits {
			tmp, _ := strconv.Atoi(s)
			sum += tmp
		}
		fmt.Println(sum)
	}
}

func main2() {
	var a, b, n int
	fmt.Scan(&n)
	for i := 0; i < n; i++ {
		fmt.Scan(&a, &b)
		fmt.Println(a + b)
	}
}

func main3() {
	input := bufio.NewScanner(os.Stdin)
	input.Scan() //读取一行内容
	t, _ := strconv.Atoi(strings.Split(input.Text(), " ")[0])
	for i := 0; i < t; i++ {
		input.Scan()
		a, _ := strconv.Atoi(strings.Split(input.Text(), " ")[0])
		b, _ := strconv.Atoi(strings.Split(input.Text(), " ")[1])
		fmt.Println(a + b)
	}
}
