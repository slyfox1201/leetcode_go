package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func main() {
	input := bufio.NewScanner(os.Stdin)

	// 如果一行过长，记得手动分配缓冲
	buf := make([]byte, 2000*1024)
	input.Buffer(buf, len(buf))

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
