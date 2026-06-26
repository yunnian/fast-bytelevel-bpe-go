package main

import (
	"reflect"
	"testing"
)

func TestSplitRegexLikeGPT2Whitespace(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{"\n-", []string{"\n", "-"}},
		{" -", []string{" -"}},
		{"  word", []string{" ", " word"}},
		{" \nword", []string{" ", "\n", "word"}},
		{"\t-", []string{"\t", "-"}},
		{"\n---", []string{"\n", "---"}},
		{" ---", []string{" ---"}},
	}
	for _, tt := range tests {
		got := splitRegexLikeGPT2(tt.input)
		if !reflect.DeepEqual(got, tt.want) {
			t.Fatalf("splitRegexLikeGPT2(%q) = %#v, want %#v", tt.input, got, tt.want)
		}
	}
}

func TestSplitRegexLikeGPT2Contractions(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{"I'm", []string{"I", "'m"}},
		{"you're", []string{"you", "'re"}},
		{"I'M", []string{"I", "'", "M"}},
	}
	for _, tt := range tests {
		got := splitRegexLikeGPT2(tt.input)
		if !reflect.DeepEqual(got, tt.want) {
			t.Fatalf("splitRegexLikeGPT2(%q) = %#v, want %#v", tt.input, got, tt.want)
		}
	}
}

func TestExtractStringField(t *testing.T) {
	line := []byte(`{"id":1,"text":"hello\nworld","other":"ignored"}`)
	got, ok := extractStringField(line, "text")
	if !ok {
		t.Fatal("extractStringField failed")
	}
	if got != "hello\nworld" {
		t.Fatalf("got %q", got)
	}
}
