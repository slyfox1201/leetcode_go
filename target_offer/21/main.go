package main

func exchange(nums []int) []int {
	for s, f := 0, 0; f < len(nums); f++ {
		if nums[f]&1 == 1 {
			nums[s], nums[f] = nums[f], nums[s]
			s++
		}
	}
	return nums
}
