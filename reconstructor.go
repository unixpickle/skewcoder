package skewcoder

import (
	"fmt"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/rnn"
)

func init() {
	var r Reconstructor
	serializer.RegisterTypedDeserializer(r.SerializerType(), DeserializeReconstructor)
}

// A Reconstructor is a layer which works by applying an
// RNN component-wise to an input feature vector.
type Reconstructor struct {
	Block rnn.Block
}

// DeserializeReconstructor deserializes a Reconstructor.
func DeserializeReconstructor(d []byte) (*Reconstructor, error) {
	block, err := serializer.DeserializeWithType(d)
	if err != nil {
		return nil, err
	}
	if b, ok := block.(rnn.Block); ok {
		return &Reconstructor{Block: b}, nil
	}
	return nil, fmt.Errorf("type is not an rnn.Block: %T", block)
}

// Apply applies the layer to an input.
func (r *Reconstructor) Apply(in autofunc.Result) autofunc.Result {
	return r.Batch(in, 1)
}

// ApplyR applies the layer to an input.
func (r *Reconstructor) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	return r.BatchR(rv, in, 1)
}

// Batch applies the layer to a batch of inputs.
func (r *Reconstructor) Batch(in autofunc.Result, n int) autofunc.Result {
	timesteps := len(in.Output()) / n
	inStates := make([]rnn.State, n)
	for i := range inStates {
		inStates[i] = r.Block.StartState()
	}
	res := &reconstructorOutput{
		n:           n,
		block:       r.Block,
		startStates: inStates,
	}
	for t := 0; t < timesteps; t++ {
		var inputs []autofunc.Result
		for i := 0; i < n; i++ {
			inputs = append(inputs, autofunc.Slice(in, i*timesteps+t, i*timesteps+t+1))
		}
		blockRes := r.Block.ApplyBlock(inStates, inputs)
		inStates = blockRes.States()
		res.outputs = append(res.outputs, blockRes)
	}
	var resVec linalg.Vector
	for _, v := range res.outputs[len(res.outputs)-1].Outputs() {
		resVec = append(resVec, v...)
	}
	res.joinedOut = resVec
	return res
}

// BatchR applies the layer to a batch of inputs.
func (r *Reconstructor) BatchR(rv autofunc.RVector, in autofunc.RResult, n int) autofunc.RResult {
	timesteps := len(in.Output()) / n
	inStates := make([]rnn.RState, n)
	for i := range inStates {
		inStates[i] = r.Block.StartRState(rv)
	}
	res := &reconstructorROutput{
		n:           n,
		block:       r.Block,
		startStates: inStates,
	}
	for t := 0; t < timesteps; t++ {
		var inputs []autofunc.RResult
		for i := 0; i < n; i++ {
			inputs = append(inputs, autofunc.SliceR(in, i*timesteps+t, i*timesteps+t+1))
		}
		blockRes := r.Block.ApplyBlockR(rv, inStates, inputs)
		inStates = blockRes.RStates()
		res.outputs = append(res.outputs, blockRes)
	}
	var resVec, resVecR linalg.Vector
	rOuts := res.outputs[len(res.outputs)-1].ROutputs()
	for i, v := range res.outputs[len(res.outputs)-1].Outputs() {
		resVec = append(resVec, v...)
		resVecR = append(resVecR, rOuts[i]...)
	}
	res.joinedOut = resVec
	res.joinedOutR = resVecR
	return res
}

// Parameters returns the parameters of the underlying
// block if it is an sgd.Learner.
func (r *Reconstructor) Parameters() []*autofunc.Variable {
	if l, ok := r.Block.(sgd.Learner); ok {
		return l.Parameters()
	} else {
		return nil
	}
}

// SerializerType returns the unique ID used to serialize
// a Reconstructor with the serializer package.
func (r *Reconstructor) SerializerType() string {
	return "github.com/unixpickle/skewcoder.Reconstructor"
}

// Serialize serializes the block if possible.
func (r *Reconstructor) Serialize() ([]byte, error) {
	if s, ok := r.Block.(serializer.Serializer); ok {
		return serializer.SerializeWithType(s)
	}
	return nil, fmt.Errorf("type is not serializer: %T", r.Block)
}

type reconstructorOutput struct {
	n           int
	block       rnn.Block
	startStates []rnn.State
	outputs     []rnn.BlockResult
	joinedOut   linalg.Vector
}

func (r *reconstructorOutput) Output() linalg.Vector {
	return r.joinedOut
}

func (r *reconstructorOutput) Constant(g autofunc.Gradient) bool {
	return false
}

func (r *reconstructorOutput) PropagateGradient(u linalg.Vector, g autofunc.Gradient) {
	blockUpstreams := make([]linalg.Vector, r.n)
	segLen := len(u) / r.n
	for i := range blockUpstreams {
		blockUpstreams[i] = u[i*segLen : (i+1)*segLen]
	}

	var downStates []rnn.StateGrad
	for t := len(r.outputs) - 1; t >= 0; t-- {
		downStates = r.outputs[t].PropagateGradient(blockUpstreams, downStates, g)
		blockUpstreams = nil
	}
	r.block.PropagateStart(r.startStates, downStates, g)
}

type reconstructorROutput struct {
	n           int
	block       rnn.Block
	startStates []rnn.RState
	outputs     []rnn.BlockRResult
	joinedOut   linalg.Vector
	joinedOutR  linalg.Vector
}

func (r *reconstructorROutput) Output() linalg.Vector {
	return r.joinedOut
}

func (r *reconstructorROutput) ROutput() linalg.Vector {
	return r.joinedOutR
}

func (r *reconstructorROutput) Constant(rg autofunc.RGradient, g autofunc.Gradient) bool {
	return false
}

func (r *reconstructorROutput) PropagateRGradient(u, uR linalg.Vector, rg autofunc.RGradient,
	g autofunc.Gradient) {
	blockUpstreams := make([]linalg.Vector, r.n)
	blockUpstreamsR := make([]linalg.Vector, r.n)
	segLen := len(u) / r.n
	for i := range blockUpstreams {
		blockUpstreams[i] = u[i*segLen : (i+1)*segLen]
		blockUpstreamsR[i] = uR[i*segLen : (i+1)*segLen]
	}

	var downStates []rnn.RStateGrad
	for t := len(r.outputs) - 1; t >= 0; t-- {
		downStates = r.outputs[t].PropagateRGradient(blockUpstreams, blockUpstreamsR,
			downStates, rg, g)
		blockUpstreams = nil
		blockUpstreamsR = nil
	}
	r.block.PropagateStartR(r.startStates, downStates, rg, g)
}
