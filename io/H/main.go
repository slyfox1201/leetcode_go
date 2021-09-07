package main

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strings"
)

func main1() {
	n := 0
	fmt.Scan(&n)
	reader := bufio.NewReader(os.Stdin)
	s, _ := reader.ReadString('\n')
	str := strings.Fields(s)
	sort.Strings(str)
	s = ""
	for _, v := range str {
		s += v + " "
	}
	fmt.Println(s[:len(s)-1])
}

func main2() {
	var n int
	fmt.Scan(&n)
	input := bufio.NewScanner(os.Stdin)
	for input.Scan() {
		s := input.Text()
		words := strings.Fields(s) // 和splits功能大体相同，有些细节不同，如/t /n等也会被分割
		sort.Strings(words)
		for i := 0; i < len(words)-1; i++ {
			fmt.Printf("%v ", words[i])
		}
		fmt.Println(words[len(words)-1])
	}
}

func main3() {
	input := bufio.NewScanner(os.Stdin)
	input.Scan()
	input.Scan()
	str := strings.Split(input.Text(), " ")
	sort.StringSlice.Sort(str)
	res := ""
	for _, v := range str {
		res = res + " " + v
	}
	fmt.Println(res[1:])
}
