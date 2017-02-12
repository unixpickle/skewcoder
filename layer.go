package skewcoder

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	var l Layer
	serializer.RegisterTypedDeserializer(l.SerializerType(), DeserializeLayer)
}

// A Layer takes inputs and feeds them to an RNN component
// by component, returning the output from the last
// timestep.
type Layer struct {
	Block anyrnn.Block
}

// DeserializeLayer deserializes a Layer.
func DeserializeLayer(d []byte) (*Layer, error) {
	var res Layer
	if err := serializer.DeserializeAny(d, &res.Block); err != nil {
		return nil, essentials.AddCtx("deserialize skewcoder Layer", err)
	}
	return &res, nil
}

// Apply applies the layer to an input.
func (l *Layer) Apply(in anydiff.Res, n int) anydiff.Res {
	inSeq := vecToSeq(in, n)
	return anyseq.Tail(anyrnn.Map(inSeq, l.Block))
}

// Parameters returns the parameters of the underlying
// block if it has any.
func (l *Layer) Parameters() []*anydiff.Var {
	if p, ok := l.Block.(anynet.Parameterizer); ok {
		return p.Parameters()
	} else {
		return nil
	}
}

// SerializerType returns the unique ID used to serialize
// a Layer with the serializer package.
func (l *Layer) SerializerType() string {
	return "github.com/unixpickle/skewcoder.Layer"
}

// Serialize serializes the block if possible.
func (l *Layer) Serialize() (d []byte, err error) {
	defer essentials.AddCtxTo("serialize skewcoder Layer", &err)
	return serializer.SerializeAny(l.Block)
}
