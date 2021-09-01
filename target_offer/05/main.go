package main

import "strings"

func replaceSpace1(s string) string {
	return strings.Join(strings.Split(s, " "), "%20")
}

func replaceSpace2(s string) string {
	res := strings.ReplaceAll(s, " ", "%20")
	return res
}

func replaceSpace3(s string) string {
	ans := ""
	for i := 0; i < len(s); i++ {
		if s[i] == ' ' {
			ans += "%20"
		} else {
			ans += string(s[i])
		}
	}
	return ans
}

func replaceSpace(s string) string {
	var res strings.Builder
	for i := range s {
		if s[i] == ' ' {
			res.WriteString("%20")
		} else {
			res.WriteByte(s[i])
		}
	}
	return res.String()
}
