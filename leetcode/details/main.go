package main

import (
	"fmt"
	"math"
	"math/bits"
	"time"
	"unsafe"
)

// 字符串操作请使用[]byte，而不是直接操作字符串
func stringDemo() {
	t1 := time.Now()
	s1 := ""
	for i := 0; i < 100000; i++ {
		s1 += "a"
	}
	elapsed1 := time.Since(t1)
	fmt.Println("app1 run time", elapsed1)

	t2 := time.Now()
	s2 := make([]byte, 0)
	for i := 0; i < 100000; i++ {
		s2 = append(s2, "a"...)
	}
	_ = string(s2)
	elapsed2 := time.Since(t2)
	fmt.Println("app2 run time", elapsed2)
}

// arr是值传递，值是一个地址
// 虽然slice传递了arr的地址，给形参赋值，不影响原参数的值
func sliceDemo(arr []int) {
	arr = append(arr, 0, 0, 0)
	//arr = arr[:len(arr) - 1] // 删减同理
	fmt.Println(arr)
}

func runSliceDemo() {
	arr := []int{1, 2, 3}
	fmt.Println(arr)
	sliceDemo(arr)
	fmt.Println(arr)
}

// golang的int，默认应该是int64？或者是和机器相关
func intDemo() {
	i := math.MaxInt32
	fmt.Printf("%T %d\n", i, i)   // int 2147483647
	fmt.Println(unsafe.Sizeof(i)) // 8
	i += 1
	fmt.Printf("%T %d\n", i, i)   // int 2147483648
	fmt.Println(unsafe.Sizeof(i)) // 8
}

// 在golang中，^即可用为异或，也可用为按位取反，golang中不存在'~'运算符
func xorDemo() {
	res := 0xff
	fmt.Printf("%b\n", res)
	res = ^res
	fmt.Printf("%b\n", res)
}

func main() {
	res := uint(0b01000000)
	fmt.Println(bits.TrailingZeros(res))
}
