package main

func getLeastNumbers(arr []int, k int) []int {
	return nil
}

func quickSort(arr []int, l, r int) {
	if l >= r {
		return
	}
	i, j := l, r
	for i < j {
		for i < j && arr[j] >= arr[l] {
			j--
		}
		for i < j && arr[i] <= arr[l] {
			i++
		}
		arr[i], arr[j] = arr[j], arr[i]
	}
	arr[i], arr[l] = arr[l], arr[i]
	quickSort(arr, l, i-1)
	quickSort(arr, i+1, r)
}
