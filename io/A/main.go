package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

func main1() {
	var a, b int
	for {
		_, err := fmt.Scan(&a, &b)
		if err != nil {
			break
		}
		fmt.Println(a + b)
	}
}

func main2() {
	input := bufio.NewScanner(os.Stdin)
	for input.Scan() {
		a, _ := strconv.Atoi(strings.Split(input.Text(), " ")[0])
		b, _ := strconv.Atoi(strings.Split(input.Text(), " ")[1])
		fmt.Println(a + b)
	}
}

func main3() {
	var a, b int
	for {
		if _, err := fmt.Scanln(&a, &b); err != io.EOF {
			fmt.Println(a + b)
		} else {
			break
		}
	}
}

func main4() {
	a, b := 0, 0
	for {
		_, err := fmt.Scanf("%d%d", &a, &b)
		if err == nil {
			fmt.Println(a + b)
		} else {
			if err == io.EOF {
				break
			}
		}
	}
}
