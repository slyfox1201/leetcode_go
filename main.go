package main

import (
	"fmt"
	"strings"
)

func main() {
	str := "abcdefg"
	arr := make([]byte, 0)
	for i := range str {
		arr = append(arr, str[i])
	}
	fmt.Println(string(arr))
	for i, s := range strings.Split(str, "") {
		fmt.Println(i, s)
	}
}
