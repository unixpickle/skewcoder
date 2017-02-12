package skewcoder

import (
	"strings"
	"testing"

	"github.com/unixpickle/anynet/anyrnn"
)

func TestSerializeErrCtx(t *testing.T) {
	l := &Layer{Block: &anyrnn.FuncBlock{}}
	_, err := l.Serialize()
	if !strings.HasPrefix(err.Error(), "serialize skewcoder Layer") {
		t.Error("bad error message:", err)
	}
}
