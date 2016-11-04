package main

import (
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"io/ioutil"
	"log"
	"os"

	"github.com/unixpickle/weakai/neuralnet"
)

func main() {
	if len(os.Args) != 3 {
		die("Usage:", os.Args[0], "<net_file> <output.gif>")
	}
	netData, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		die("Read net:", err)
	}
	net, err := neuralnet.DeserializeNetwork(netData)
	if err != nil {
		die("Deserialize net:", err)
	}

	grid := NewGrid(net)

	log.Println("Creating frames...")

	p := gridPalette()
	res := gif.GIF{}
	for i := 0; i <= 20; i++ {
		var frame image.Image
		if i <= 10 {
			frame = grid.Frame(float64(i) / 10)
		} else {
			frame = grid.Frame(1 - float64(i-10)/10)
		}
		res.Image = append(res.Image, paletted(frame, p))
		if i%10 == 0 {
			res.Delay = append(res.Delay, 70)
		} else {
			res.Delay = append(res.Delay, 10)
		}
	}

	log.Println("Saving result...")
	f, err := os.Create(os.Args[2])
	if err != nil {
		die(err)
	}
	defer f.Close()
	if err := gif.EncodeAll(f, &res); err != nil {
		fmt.Fprintln(os.Stderr, "Encode failed:", err)
	}
}

func gridPalette() color.Palette {
	res := make(color.Palette, 0x100)
	for i := range res {
		res[i] = color.RGBA{
			R: uint8(i),
			G: uint8(i),
			B: uint8(i),
			A: 0xff,
		}
	}
	return res
}

func paletted(img image.Image, p color.Palette) *image.Paletted {
	res := image.NewPaletted(img.Bounds(), p)
	for y := 0; y < img.Bounds().Dy(); y++ {
		for x := 0; x < img.Bounds().Dx(); x++ {
			res.Set(x, y, img.At(x, y))
		}
	}
	return res
}

func die(args ...interface{}) {
	fmt.Fprintln(os.Stderr, args...)
	os.Exit(1)
}
