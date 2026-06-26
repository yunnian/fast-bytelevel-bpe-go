package main

import (
	"bufio"
	"bytes"
	"container/heap"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"
)

var byteToRune = bytesToUnicode()
var contractionSuffixes = []string{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"}

type Pair uint64

func makePair(a, b uint32) Pair { return Pair(uint64(a)<<32 | uint64(b)) }
func left(p Pair) uint32        { return uint32(uint64(p) >> 32) }
func right(p Pair) uint32       { return uint32(uint64(p)) }

type Word struct {
	Tokens []uint32
	Count  int64
}

type Delta struct {
	Pair  Pair
	Delta int64
}

type HeapItem struct {
	Pair  Pair
	Count int64
}

type PairHeap []HeapItem

func (h PairHeap) Len() int { return len(h) }
func (h PairHeap) Less(i, j int) bool {
	if h[i].Count != h[j].Count {
		return h[i].Count > h[j].Count
	}
	return h[i].Pair < h[j].Pair
}
func (h PairHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *PairHeap) Push(x any)   { *h = append(*h, x.(HeapItem)) }
func (h *PairHeap) Pop() any     { old := *h; x := old[len(old)-1]; *h = old[:len(old)-1]; return x }

type MergeRecord struct {
	Pair  Pair
	NewID uint32
}

type MergeBatchResult struct {
	DeltaCounts  map[Pair]int64
	PairWordSeen map[Pair][]int
	ChangedWords int
}

type Trainer struct {
	VocabSize     int
	MinFrequency  int64
	DebugNewID    int
	SpecialTokens []string
	Words         []Word

	PairCounts  map[Pair]int64
	PairToWords map[Pair][]int
	Queue       PairHeap

	IDToToken []string
	TokenToID map[string]uint32
	Merges    []MergeRecord
}

type stringListFlag []string

func (s *stringListFlag) String() string { return strings.Join(*s, ",") }
func (s *stringListFlag) Set(value string) error {
	*s = append(*s, value)
	return nil
}

func main() {
	input := flag.String("input", "", "JSONL input path")
	field := flag.String("field", "text", "JSONL field to train on")
	maxTexts := flag.Int("max-texts", 0, "maximum non-empty JSONL text records; 0 means no limit")
	vocabSize := flag.Int("vocab-size", 32779, "total HF vocab size, including special tokens")
	minFrequency := flag.Int64("min-frequency", 2, "minimum pair frequency")
	out := flag.String("out", "tokenizer.json", "output HF-compatible tokenizer JSON path")
	progressEvery := flag.Int("progress-every", 500, "print progress every N merges")
	debugNewID := flag.Int("debug-new-id", 0, "print top pair counts before creating this vocab id")
	var specialTokens stringListFlag
	flag.Var(&specialTokens, "special-token", "special token to reserve; may be repeated")
	flag.Parse()
	if *input == "" {
		fmt.Fprintln(os.Stderr, "--input is required")
		os.Exit(2)
	}

	start := time.Now()
	fmt.Printf("GO_HF_START %s\n", start.Format("2006-01-02 15:04:05"))
	fmt.Printf("go=%s cpus=%d input=%s field=%s max_texts=%d vocab_size=%d min_frequency=%d special_tokens=%d\n",
		runtime.Version(), runtime.NumCPU(), *input, *field, *maxTexts, *vocabSize, *minFrequency, len(specialTokens))

	tr := NewTrainer(*vocabSize, *minFrequency, specialTokens)
	tr.DebugNewID = *debugNewID
	chunkCounts, err := readChunkCounts(*input, *field, *maxTexts)
	if err != nil {
		panic(err)
	}
	fmt.Printf("unique_chunks=%d elapsed=%.2fs\n", len(chunkCounts), time.Since(start).Seconds())

	tr.LoadChunks(chunkCounts)
	fmt.Printf("words=%d elapsed=%.2fs\n", len(tr.Words), time.Since(start).Seconds())

	tr.InitPairs()
	fmt.Printf("initial_pairs=%d elapsed=%.2fs\n", len(tr.PairCounts), time.Since(start).Seconds())

	tr.Train(start, *progressEvery)
	if err := tr.Save(*out); err != nil {
		panic(err)
	}
	fmt.Printf("saved=%s elapsed=%.2fs\n", *out, time.Since(start).Seconds())
	fmt.Printf("GO_HF_END %s\n", time.Now().Format("2006-01-02 15:04:05"))
}

func NewTrainer(vocabSize int, minFrequency int64, specialTokens []string) *Trainer {
	tr := &Trainer{
		VocabSize:     vocabSize,
		MinFrequency:  minFrequency,
		SpecialTokens: append([]string(nil), specialTokens...),
		PairCounts:    make(map[Pair]int64),
		PairToWords:   make(map[Pair][]int),
		IDToToken:     make([]string, 0, vocabSize),
		TokenToID:     make(map[string]uint32, vocabSize),
		Merges:        make([]MergeRecord, 0, vocabSize),
	}
	for _, token := range tr.SpecialTokens {
		tr.addToken(token)
	}
	alphabet := make([]rune, 0, 256)
	for _, r := range byteToRune {
		alphabet = append(alphabet, r)
	}
	sort.Slice(alphabet, func(i, j int) bool { return alphabet[i] < alphabet[j] })
	for _, r := range alphabet {
		tr.addToken(string(r))
	}
	return tr
}

func (t *Trainer) addToken(token string) uint32 {
	if id, ok := t.TokenToID[token]; ok {
		return id
	}
	id := uint32(len(t.IDToToken))
	t.IDToToken = append(t.IDToToken, token)
	t.TokenToID[token] = id
	return id
}

func bytesToUnicode() [256]rune {
	var out [256]rune
	bs := make([]int, 0, 256)
	for b := int('!'); b <= int('~'); b++ {
		bs = append(bs, b)
	}
	for b := 0xA1; b <= 0xAC; b++ {
		bs = append(bs, b)
	}
	for b := 0xAE; b <= 0xFF; b++ {
		bs = append(bs, b)
	}
	seen := make(map[int]bool, 256)
	for _, b := range bs {
		seen[b] = true
		out[b] = rune(b)
	}
	n := 0
	for b := 0; b < 256; b++ {
		if !seen[b] {
			out[b] = rune(256 + n)
			n++
		}
	}
	return out
}

func readChunkCounts(path string, field string, maxTexts int) (map[string]int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	counts := make(map[string]int64, 1_000_000)
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 64*1024*1024)
	n := 0
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		text, ok := extractStringField(line, field)
		if !ok || text == "" {
			continue
		}
		for _, chunk := range splitRegexLikeGPT2(text) {
			if chunk != "" {
				counts[byteLevelEncode(chunk)]++
			}
		}
		n++
		if maxTexts > 0 && n >= maxTexts {
			break
		}
	}
	return counts, scanner.Err()
}

func extractStringField(line []byte, field string) (string, bool) {
	fieldKey := []byte(strconv.Quote(field))
	search := line
	offset := 0
	for {
		i := bytes.Index(search, fieldKey)
		if i < 0 {
			return "", false
		}
		i += offset + len(fieldKey)
		for i < len(line) && (line[i] == ' ' || line[i] == '\t' || line[i] == '\r' || line[i] == '\n') {
			i++
		}
		if i >= len(line) || line[i] != ':' {
			offset = i
			search = line[offset:]
			continue
		}
		i++
		for i < len(line) && (line[i] == ' ' || line[i] == '\t' || line[i] == '\r' || line[i] == '\n') {
			i++
		}
		if i >= len(line) || line[i] != '"' {
			return "", false
		}
		start := i
		escaped := false
		for i = start + 1; i < len(line); i++ {
			switch line[i] {
			case '\\':
				escaped = true
				i++
			case '"':
				if !escaped {
					return string(line[start+1 : i]), true
				}
				text, err := strconv.Unquote(string(line[start : i+1]))
				return text, err == nil
			}
		}
		return "", false
	}
}

func splitRegexLikeGPT2(s string) []string {
	out := make([]string, 0, 64)
	for len(s) > 0 {
		if c, ok := lowerContraction(s); ok {
			out = append(out, c)
			s = s[len(c):]
			continue
		}

		r, size := utf8.DecodeRuneInString(s)
		if r == utf8.RuneError && size == 1 {
			out = append(out, s[:1])
			s = s[1:]
			continue
		}

		if unicode.IsSpace(r) {
			runEnd := size
			for runEnd < len(s) {
				rr, ss := utf8.DecodeRuneInString(s[runEnd:])
				if !unicode.IsSpace(rr) {
					break
				}
				runEnd += ss
			}
			if runEnd == len(s) {
				out = append(out, s)
				break
			}
			if runEnd > size {
				lastStart := 0
				for i := range s[:runEnd] {
					lastStart = i
				}
				out = append(out, s[:lastStart])
				s = s[lastStart:]
				continue
			}
			if r != ' ' {
				out = append(out, s[:size])
				s = s[size:]
				continue
			}
		}

		prefix := 0
		if r == ' ' && len(s) > size {
			prefix = size
			r, size = utf8.DecodeRuneInString(s[prefix:])
		}

		switch {
		case unicode.IsLetter(r):
			i := prefix + size
			for i < len(s) {
				rr, ss := utf8.DecodeRuneInString(s[i:])
				if !unicode.IsLetter(rr) {
					break
				}
				i += ss
			}
			out = append(out, s[:i])
			s = s[i:]
		case unicode.IsNumber(r):
			i := prefix + size
			for i < len(s) {
				rr, ss := utf8.DecodeRuneInString(s[i:])
				if !unicode.IsNumber(rr) {
					break
				}
				i += ss
			}
			out = append(out, s[:i])
			s = s[i:]
		default:
			i := prefix + size
			for i < len(s) {
				rr, ss := utf8.DecodeRuneInString(s[i:])
				if unicode.IsSpace(rr) || unicode.IsLetter(rr) || unicode.IsNumber(rr) {
					break
				}
				i += ss
			}
			out = append(out, s[:i])
			s = s[i:]
		}
	}
	return out
}

func lowerContraction(s string) (string, bool) {
	for _, c := range contractionSuffixes {
		if strings.HasPrefix(s, c) {
			return c, true
		}
	}
	return "", false
}

func byteLevelEncode(s string) string {
	runes := make([]rune, 0, len(s))
	for _, b := range []byte(s) {
		runes = append(runes, byteToRune[b])
	}
	return string(runes)
}

func (t *Trainer) LoadChunks(chunkCounts map[string]int64) {
	t.Words = make([]Word, 0, len(chunkCounts))
	for chunk, count := range chunkCounts {
		tokens := make([]uint32, 0, len(chunk))
		for _, r := range chunk {
			tokens = append(tokens, t.TokenToID[string(r)])
		}
		t.Words = append(t.Words, Word{Tokens: tokens, Count: count})
	}
}

func (t *Trainer) InitPairs() {
	for wordID := range t.Words {
		w := &t.Words[wordID]
		seen := make(map[Pair]struct{}, 8)
		for i := 0; i+1 < len(w.Tokens); i++ {
			p := makePair(w.Tokens[i], w.Tokens[i+1])
			t.PairCounts[p] += w.Count
			seen[p] = struct{}{}
		}
		for p := range seen {
			t.PairToWords[p] = append(t.PairToWords[p], wordID)
		}
	}
	t.Queue = make(PairHeap, 0, len(t.PairCounts))
	for p, c := range t.PairCounts {
		if c > 0 {
			t.Queue = append(t.Queue, HeapItem{Pair: p, Count: c})
		}
	}
	heap.Init(&t.Queue)
}

func (t *Trainer) Train(start time.Time, progressEvery int) {
	for len(t.IDToToken) < t.VocabSize {
		if t.DebugNewID > 0 && len(t.IDToToken) == t.DebugNewID {
			t.PrintTopPairs(20)
		}
		top, ok := t.popValid()
		if !ok || top.Count < t.MinFrequency {
			break
		}

		newToken := t.IDToToken[left(top.Pair)] + t.IDToToken[right(top.Pair)]
		newID := t.addToken(newToken)
		t.Merges = append(t.Merges, MergeRecord{Pair: top.Pair, NewID: newID})

		affected := t.PairToWords[top.Pair]
		delete(t.PairToWords, top.Pair)

		result := t.MergeAffected(affected, top.Pair, newID)

		for p, d := range result.DeltaCounts {
			c := t.PairCounts[p] + d
			if c <= 0 {
				delete(t.PairCounts, p)
				continue
			}
			t.PairCounts[p] = c
			heap.Push(&t.Queue, HeapItem{Pair: p, Count: c})
		}
		for p, words := range result.PairWordSeen {
			t.PairToWords[p] = append(t.PairToWords[p], words...)
		}

		if progressEvery > 0 && (int(newID) < 300 || int(newID)%progressEvery == 0) {
			fmt.Printf("merge %d: (%s,%s) count=%d affected=%d changed=%d pairs=%d heap=%d elapsed=%.2fs\n",
				newID, t.IDToToken[left(top.Pair)], t.IDToToken[right(top.Pair)], top.Count, len(affected), result.ChangedWords, len(t.PairCounts), t.Queue.Len(), time.Since(start).Seconds())
		}
	}
}

func (t *Trainer) MergeAffected(affected []int, pair Pair, newID uint32) MergeBatchResult {
	unique := make([]int, 0, len(affected))
	wordSeen := make(map[int]struct{}, len(affected))
	for _, wordID := range affected {
		if _, ok := wordSeen[wordID]; ok {
			continue
		}
		wordSeen[wordID] = struct{}{}
		unique = append(unique, wordID)
	}
	if len(unique) < 20000 || runtime.NumCPU() <= 1 {
		return t.mergeAffectedRange(unique, pair, newID)
	}

	workers := runtime.NumCPU()
	if len(unique)/workers < 5000 {
		workers = max(1, len(unique)/5000)
	}
	chunkSize := (len(unique) + workers - 1) / workers
	results := make([]MergeBatchResult, workers)
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		start := worker * chunkSize
		end := start + chunkSize
		if end > len(unique) {
			end = len(unique)
		}
		if start >= end {
			results = results[:worker]
			break
		}
		wg.Add(1)
		go func(i, start, end int) {
			defer wg.Done()
			results[i] = t.mergeAffectedRange(unique[start:end], pair, newID)
		}(worker, start, end)
	}
	wg.Wait()

	out := MergeBatchResult{
		DeltaCounts:  make(map[Pair]int64, 256),
		PairWordSeen: make(map[Pair][]int, 256),
	}
	for _, result := range results {
		out.ChangedWords += result.ChangedWords
		for p, d := range result.DeltaCounts {
			out.DeltaCounts[p] += d
		}
		for p, words := range result.PairWordSeen {
			out.PairWordSeen[p] = append(out.PairWordSeen[p], words...)
		}
	}
	return out
}

func (t *Trainer) mergeAffectedRange(wordIDs []int, pair Pair, newID uint32) MergeBatchResult {
	out := MergeBatchResult{
		DeltaCounts:  make(map[Pair]int64, 64),
		PairWordSeen: make(map[Pair][]int, 64),
	}
	for _, wordID := range wordIDs {
		deltas, changed := t.Words[wordID].Merge(pair, newID)
		if !changed {
			continue
		}
		out.ChangedWords++
		mult := t.Words[wordID].Count
		for _, d := range deltas {
			out.DeltaCounts[d.Pair] += d.Delta * mult
			if d.Delta > 0 {
				out.PairWordSeen[d.Pair] = append(out.PairWordSeen[d.Pair], wordID)
			}
		}
	}
	return out
}

func (t *Trainer) PrintTopPairs(limit int) {
	items := make([]HeapItem, 0, len(t.PairCounts))
	for p, c := range t.PairCounts {
		if c > 0 {
			items = append(items, HeapItem{Pair: p, Count: c})
		}
	}
	sort.Slice(items, func(i, j int) bool {
		if items[i].Count != items[j].Count {
			return items[i].Count > items[j].Count
		}
		return items[i].Pair < items[j].Pair
	})
	if len(items) < limit {
		limit = len(items)
	}
	fmt.Printf("DEBUG_TOP_PAIRS new_id=%d pairs=%d\n", len(t.IDToToken), len(items))
	for i := 0; i < limit; i++ {
		p := items[i].Pair
		fmt.Printf("  #%d count=%d pair=(%d,%d) token=(%s,%s)\n",
			i+1, items[i].Count, left(p), right(p), t.IDToToken[left(p)], t.IDToToken[right(p)])
	}
}

func (t *Trainer) popValid() (HeapItem, bool) {
	for t.Queue.Len() > 0 {
		item := heap.Pop(&t.Queue).(HeapItem)
		if t.PairCounts[item.Pair] == item.Count {
			return item, true
		}
	}
	return HeapItem{}, false
}

func (w *Word) Merge(pair Pair, newID uint32) ([]Delta, bool) {
	a, b := left(pair), right(pair)
	old := w.Tokens
	newTokens := make([]uint32, 0, len(old))
	oldTouchedIdx := make(map[int]struct{}, 8)
	newTouched := make(map[Pair]int64, 8)
	changed := false

	i := 0
	for i < len(old) {
		if i+1 < len(old) && old[i] == a && old[i+1] == b {
			changed = true
			if i > 0 {
				oldTouchedIdx[i-1] = struct{}{}
			}
			oldTouchedIdx[i] = struct{}{}
			if i+2 < len(old) {
				oldTouchedIdx[i+1] = struct{}{}
			}
			newTokens = append(newTokens, newID)
			i += 2
			continue
		}
		newTokens = append(newTokens, old[i])
		i++
	}
	if !changed {
		return nil, false
	}

	for i := 0; i+1 < len(newTokens); i++ {
		if newTokens[i] == newID || newTokens[i+1] == newID {
			newTouched[makePair(newTokens[i], newTokens[i+1])]++
		}
	}

	deltas := make([]Delta, 0, len(oldTouchedIdx)+len(newTouched))
	for idx := range oldTouchedIdx {
		deltas = append(deltas, Delta{Pair: makePair(old[idx], old[idx+1]), Delta: -1})
	}
	for p, c := range newTouched {
		deltas = append(deltas, Delta{Pair: p, Delta: c})
	}
	w.Tokens = newTokens
	return deltas, true
}

func (t *Trainer) Save(path string) error {
	vocab := make(map[string]uint32, len(t.IDToToken))
	for id, token := range t.IDToToken {
		vocab[token] = uint32(id)
	}
	merges := make([][2]string, 0, len(t.Merges))
	for _, m := range t.Merges {
		merges = append(merges, [2]string{t.IDToToken[left(m.Pair)], t.IDToToken[right(m.Pair)]})
	}
	added := make([]map[string]any, 0, len(t.SpecialTokens))
	for i, token := range t.SpecialTokens {
		added = append(added, map[string]any{
			"id": i, "content": token, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true,
		})
	}
	payload := map[string]any{
		"version":        "1.0",
		"truncation":     nil,
		"padding":        nil,
		"added_tokens":   added,
		"normalizer":     nil,
		"pre_tokenizer":  map[string]any{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
		"post_processor": nil,
		"decoder":        map[string]any{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
		"model": map[string]any{
			"type": "BPE", "dropout": nil, "unk_token": nil, "continuing_subword_prefix": nil, "end_of_word_suffix": nil,
			"fuse_unk": false, "byte_fallback": false, "ignore_merges": false, "vocab": vocab, "merges": merges,
		},
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(payload)
}
