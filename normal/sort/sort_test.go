package sort

import (
	"testing"
)

var target = []int{0, 1, 2, 3, 4, 5}

func TestBubbleSort(t *testing.T) {
	arr := []int{3, 5, 2, 1, 4, 0}
	BubbleSort(arr)
	if !sliceEqual(arr, target) {
		t.Errorf("%+v is not sorted.", arr)
	}
}

func TestSelectionSort(t *testing.T) {
	arr := []int{3, 5, 2, 1, 4, 0}
	SelectionSort(arr)
	if !sliceEqual(arr, target) {
		t.Errorf("%+v is not sorted.", arr)
	}
}

func TestInsertionSort(t *testing.T) {
	arr := []int{3, 5, 2, 1, 4, 0}
	InsertionSort(arr)
	if !sliceEqual(arr, target) {
		t.Errorf("%+v is not sorted.", arr)
	}
}

func TestShellSort(t *testing.T) {
	arr := []int{3, 5, 2, 1, 4, 0}
	ShellSort(arr)
	if !sliceEqual(arr, target) {
		t.Errorf("%+v is not sorted.", arr)
	}
}

func TestMergeSort(t *testing.T) {
	arr := []int{3, 5, 2, 1, 4, 0}
	MergeSort(arr)
	if !sliceEqual(arr, target) {
		t.Errorf("%+v is not sorted.", arr)
	}
}

func TestQuickSort(t *testing.T) {
	arr := []int{3, 5, 2, 1, 4, 0}
	QuickSort(arr)
	if !sliceEqual(arr, target) {
		t.Errorf("%+v is not sorted.", arr)
	}
}

func TestHeapSort(t *testing.T) {
	arr := []int{3, 5, 2, 1, 4, 0}
	HeapSort(arr)
	if !sliceEqual(arr, target) {
		t.Errorf("%+v is not sorted.", arr)
	}
}

func sliceEqual(arr1, arr2 []int) bool {
	if len(arr1) != len(arr2) {
		return false
	}
	for i := 0; i < len(arr1); i++ {
		if arr1[i] != arr2[i] {
			return false
		}
	}
	return true
}
