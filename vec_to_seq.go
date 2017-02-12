package skewcoder

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
)

type vecSeqResult struct {
	In  anydiff.Res
	Out []*anyseq.Batch
}

func vecToSeq(v anydiff.Res, batchSize int) anyseq.Seq {
	return &vecSeqResult{In: v, Out: vecToBatches(v.Output(), batchSize)}
}

func (v *vecSeqResult) Creator() anyvec.Creator {
	return v.In.Output().Creator()
}

func (v *vecSeqResult) Output() []*anyseq.Batch {
	return v.Out
}

func (v *vecSeqResult) Vars() anydiff.VarSet {
	return v.In.Vars()
}

func (v *vecSeqResult) Propagate(u []*anyseq.Batch, g anydiff.Grad) {
	vecUp := batchesToVec(v.Creator(), u)
	v.In.Propagate(vecUp, g)
}

func vecToBatches(vec anyvec.Vector, batchSize int) []*anyseq.Batch {
	if vec.Len()%batchSize != 0 {
		panic("batch size must divide vector length")
	}

	seqLen := vec.Len() / batchSize
	inMat := &anyvec.Matrix{Data: vec, Rows: batchSize, Cols: seqLen}
	trans := &anyvec.Matrix{Data: vec.Copy(), Rows: seqLen, Cols: batchSize}
	trans.Transpose(inMat)

	var res []*anyseq.Batch
	for i := 0; i < seqLen; i++ {
		part := trans.Data.Slice(i*batchSize, (i+1)*batchSize)
		present := make([]bool, batchSize)
		for j := range present {
			present[j] = true
		}
		res = append(res, &anyseq.Batch{Packed: part, Present: present})
	}

	return res
}

func batchesToVec(c anyvec.Creator, b []*anyseq.Batch) anyvec.Vector {
	if len(b) == 0 {
		return c.MakeVector(0)
	}
	batchSize := b[0].NumPresent()

	var parts []anyvec.Vector
	for _, batch := range b {
		parts = append(parts, batch.Packed)
	}
	joined := c.Concat(parts...)

	inMat := &anyvec.Matrix{Data: joined, Rows: len(b), Cols: batchSize}
	outMat := &anyvec.Matrix{Data: joined.Copy(), Rows: batchSize, Cols: len(b)}
	outMat.Transpose(inMat)

	return outMat.Data
}
