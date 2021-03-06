package main

import (
	"image"
	"image/color"
	"math/rand"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/mnist"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/skewcoder"
)

const GridSize = 5

type Grid struct {
	leftImages  []linalg.Vector
	rightImages []linalg.Vector
	decoder     anynet.Net
}

func NewGrid(net anynet.Net) *Grid {
	encoder, decoder := splitNet(net)
	validation := mnist.LoadTestingDataSet()
	res := &Grid{
		decoder: decoder,
	}
	for i := 0; i < GridSize*GridSize; i++ {
		res.leftImages = append(res.leftImages, randomEncImage(validation, encoder))
		res.rightImages = append(res.rightImages, randomEncImage(validation, encoder))
	}
	return res
}

func (g *Grid) Frame(t float64) image.Image {
	res := image.NewRGBA(image.Rect(0, 0, 28*GridSize*2, 28*GridSize))
	var idx int
	for y := 0; y < GridSize; y++ {
		for x := 0; x < GridSize; x++ {
			left := g.leftImages[idx]
			right := g.rightImages[idx]
			newLeft := left.Copy().Scale(1 - t).Add(right.Copy().Scale(t))
			newRight := right.Copy().Scale(1 - t).Add(left.Copy().Scale(t))
			g.drawDecoded(res, x*2, y, newLeft)
			g.drawDecoded(res, x*2+1, y, newRight)
			idx++
		}
	}
	return res
}

func (g *Grid) drawDecoded(out *image.RGBA, x, y int, vec linalg.Vector) {
	inVec := anyvec32.MakeVectorData(anyvec32.MakeNumericList(vec))
	decoded32 := g.decoder.Apply(anydiff.NewConst(inVec), 1).Output().Data().([]float32)
	decoded := to64Bit(decoded32)
	var idx int
	for subY := 0; subY < 28; subY++ {
		for subX := 0; subX < 28; subX++ {
			brightness := 1 - decoded[idx]
			idx++
			pxl := color.RGBA{
				R: uint8(brightness*0xff + 0.5),
				G: uint8(brightness*0xff + 0.5),
				B: uint8(brightness*0xff + 0.5),
				A: 0xff,
			}
			out.Set(x*28+subX, y*28+subY, pxl)
		}
	}
}

func randomEncImage(d mnist.DataSet, enc anynet.Net) linalg.Vector {
	randIdx := rand.Intn(len(d.Samples))
	sample := d.Samples[randIdx].Intensities
	inVec := anyvec32.MakeVectorData(anyvec32.MakeNumericList(sample))
	res32 := enc.Apply(anydiff.NewConst(inVec), 1).Output().Data().([]float32)
	return to64Bit(res32)
}

func splitNet(net anynet.Net) (encode, decode anynet.Net) {
	for i, x := range net {
		if _, ok := x.(*skewcoder.Layer); ok {
			return net[:i], net[i:]
		}
	}
	essentials.Die("no Reconstructor layer found")
	return nil, nil
}

func to64Bit(res32 []float32) linalg.Vector {
	res := make(linalg.Vector, len(res32))
	for i, x := range res32 {
		res[i] = float64(x)
	}
	return res
}
