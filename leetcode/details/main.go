package main

import (
	"fmt"
	"math"
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

// arr是切片结构体，即一个指向底层数组的指针，一个len，一个cap，结构体
// 是复制传递，因此只改变了形参结构体的值，原结构体未改变
func sliceDemo(arr []int) {
	//arr = append(arr, 0, 0, 0) // 此句会改变底层数组和形参结构体
	arr = arr[:len(arr) - 1] // 此句只改变形参结构体
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

// dfs对比lc47全排列与lc90子集

func main() {
	//res := uint(0b01000000)
	//fmt.Println(bits.TrailingZeros(res))
	runSliceDemo()
}
