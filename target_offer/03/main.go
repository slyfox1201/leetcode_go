package main

func findRepeatNumber1(nums []int) int {
	cache := make(map[int]bool)
	for _, n := range nums {
		if cache[n] { // 如果不存在，则返回默认的0值，这里为false。没必要写成_, ok = cache[n].
			return n
		}
		cache[n] = true
	}
	return -1
}

func findRepeatNumber(nums []int) int {
	for i := 0; i < len(nums); i++ {
		for nums[i] != i {
			if nums[i] == nums[nums[i]] {
				return nums[i]
			}
			nums[i], nums[nums[i]] = nums[nums[i]], nums[i]
		}
	}
	return -1
}
