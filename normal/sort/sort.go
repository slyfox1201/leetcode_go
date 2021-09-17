package sort

import (
	"math/rand"
	"time"
)

//冒泡排序
func BubbleSort(arr []int) {
	for i := 0; i < len(arr); i++ {
		for j := 0; j < len(arr)-1-i; j++ {
			if arr[j+1] < arr[j] {
				arr[j], arr[j+1] = arr[j+1], arr[j]
			}
		}
	}
}

// 选择排序
func SelectionSort(arr []int) {
	for i := 0; i < len(arr); i++ {
		minIndex := i
		for j := i; j < len(arr); j++ {
			if arr[j] < arr[minIndex] {
				minIndex = j
			}
		}
		arr[minIndex], arr[i] = arr[i], arr[minIndex]
	}
}

// 插入排序
func InsertionSort(arr []int) {
	var cur int
	for i := 1; i < len(arr); i++ {
		cur = arr[i]
		preIndex := i - 1
		for preIndex >= 0 && cur < arr[preIndex] {
			arr[preIndex+1] = arr[preIndex]
			preIndex--
		}
		arr[preIndex+1] = cur
	}
}

// 希尔排序(优化的插入排序)
func ShellSort(arr []int) {
	l := len(arr)
	cur, gap := 0, l/2
	for gap > 0 {
		for i := gap; i < l; i++ {
			cur = arr[i]
			preIndex := i - gap
			for preIndex >= 0 && cur < arr[preIndex] {
				arr[preIndex+gap] = arr[preIndex]
				preIndex -= gap
			}
			arr[preIndex+gap] = cur
		}
		gap /= 2
	}
}

// 归并排序
func MergeSort(arr []int) {
	mergeSort(arr, 0, len(arr)-1)
}

func mergeSort(arr []int, start, end int) {
	if start == end {
		return
	}
	mid := start + (end-start)/2
	mergeSort(arr, start, mid)
	mergeSort(arr, mid+1, end)
	merge(arr, start, mid, end)
}

func merge(arr []int, start, mid, end int) {
	left := make([]int, mid+1-start)
	right := make([]int, end-mid)

	copy(left, arr[start:mid+1])
	copy(right, arr[mid+1:end+1])
	for i, j, k := 0, 0, start; k <= end; k++ {
		if i >= len(left) {
			arr[k] = right[j]
			j++
		} else if j >= len(right) {
			arr[k] = left[i]
			i++
		} else if left[i] > right[j] {
			arr[k] = right[j]
			j++
		} else {
			arr[k] = left[i]
			i++
		}
	}
}

// 快速排序
func QuickSort(arr []int) {
	quickSort(arr, 0, len(arr)-1)
}

func quickSort(arr []int, start int, end int) {
	if start == end {
		return
	}
	smallIndex := partition(arr, start, end)
	if smallIndex > start {
		quickSort(arr, start, smallIndex-1)
	}
	if smallIndex < end {
		quickSort(arr, smallIndex+1, end)
	}
}

func partition(arr []int, start int, end int) int {
	rand.Seed(time.Now().Unix())
	pivot := start + rand.Intn(end-start+1)
	smallIndex := start - 1
	arr[pivot], arr[end] = arr[end], arr[pivot]
	for i := start; i <= end; i++ {
		if arr[i] <= arr[end] {
			smallIndex++
			if i > smallIndex {
				arr[smallIndex], arr[i] = arr[i], arr[smallIndex]
			}
		}
	}
	return smallIndex
}

// 堆排序
func HeapSort(arr []int) {
	for i := (len(arr) - 1) / 2; i >= 0; i-- {
		adjustHeap(arr, i, len(arr))
	}
	for i := len(arr) - 1; i > 0; i-- {
		arr[i], arr[0] = arr[0], arr[i]
		adjustHeap(arr, 0, i)
	}
}

func adjustHeap(arr []int, root, size int) {
	left := 2*root + 1
	right := 2*root + 2
	max := root
	if left < size && arr[left] > arr[max] {
		max = left
	}
	if right < size && arr[right] > arr[max] {
		max = right
	}
	if max != root {
		arr[root], arr[max] = arr[max], arr[root]
		adjustHeap(arr, max, size)
	}
}
