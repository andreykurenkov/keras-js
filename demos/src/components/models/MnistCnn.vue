<template>
  <div class="demo mnist-cnn">
    <div class="title">
      <span>Basic Convnet for MNIST</span>
    </div>
    <mdl-spinner v-if="modelLoading && loadingProgress < 100"></mdl-spinner>
    <div class="loading-progress" v-if="modelLoading && loadingProgress < 100">
      Loading...{{ loadingProgress }}%
    </div>
    <div class="columns input-output" v-if="!modelLoading">
      <div class="column input-column">
        <div class="input-container">
          <div class="input-label">Draw any digit (0-9) here <span class="arrow">⤸</span></div>
          <div class="canvas-container">
            <canvas
              id="input-canvas" width="240" height="240"
              @mousedown="activateDraw"
              @mouseup="deactivateDrawAndPredict"
              @mouseleave="deactivateDrawAndPredict"
              @mousemove="draw"
              @touchstart="activateDraw"
              @touchend="deactivateDrawAndPredict"
              @touchmove="draw"
            ></canvas>
            <canvas id="input-canvas-scaled" width="28" height="28" style="display:none;"></canvas>
            <canvas id="input-canvas-centercrop" style="display:none;"></canvas>
          </div>
          <div class="input-buttons">
            <div class="input-clear-button" @click="clear"><i class="material-icons">clear</i>CLEAR</div>
          </div>
        </div>
      </div>
      <div class="column is-2 controls-column">
        <mdl-switch v-model="useGpu" :disabled="modelLoading || !hasWebgl">use GPU</mdl-switch>
      </div>
      <div class="column output-column">
        <div class="output">
          <div class="output-class"
            :class="{ predicted: i === predictedClass }"
            v-for="i in outputClasses"
            :key="`output-class-${i}`"
          >
            <div class="output-label">{{ i }}</div>
            <div class="output-bar"
              :style="{ height: `${Math.round(100 * output[i])}px`, background: `rgba(27, 188, 155, ${output[i].toFixed(2)})` }"
            ></div>
          </div>
        </div>
      </div>
    </div>
    <div class="layer-results-container"  v-if="!modelLoading" id="results-container">
    	<div id="webgl_container"></div>
    </div>
  </div>
</template>

<script>
import debounce from 'lodash/debounce'
import range from 'lodash/range'
import * as utils from '../../utils'
require('../../js/controls/myOrbitControls.js')
const MODEL_FILEPATHS_DEV = {
  model: '/demos/data/mnist_cnn/mnist_cnn.json',
  weights: '/demos/data/mnist_cnn/mnist_cnn_weights.buf',
  metadata: '/demos/data/mnist_cnn/mnist_cnn_metadata.json'
}
const MODEL_FILEPATHS_PROD = {
  model: 'https://transcranial.github.io/keras-js-demos-data/mnist_cnn/mnist_cnn.json',
  weights: 'https://transcranial.github.io/keras-js-demos-data/mnist_cnn/mnist_cnn_weights.buf',
  metadata: 'https://transcranial.github.io/keras-js-demos-data/mnist_cnn/mnist_cnn_metadata.json'
}
const MODEL_CONFIG = { filepaths: process.env.NODE_ENV === 'production' ? MODEL_FILEPATHS_PROD : MODEL_FILEPATHS_DEV }

const LAYER_DISPLAY_CONFIG = {
  input: { type:'conv', numFilters: 32, filterSize: 3, stride: 1, heading: '32 3x3 filters, padding valid, 1x1 strides', scalingFactor: 2, z: -155},
  conv2d_1: { type:'conv', numFilters: 32, filterSize: 3, stride: 1, heading: '32 3x3 filters, padding valid, 1x1 strides', scalingFactor: 2, z: -155},
  activation_1: { type:'activation', heading: 'ReLU', scalingFactor: 2, z: -105 },
  conv2d_2: { type:'conv', numFilters: 32, filterSize: 3, stride: 1, heading: '32 3x3 filters, padding valid, 1x1 strides', scalingFactor: 2, z: -55 },
  activation_2: { type:'activation',heading: 'ReLU', scalingFactor: 2, z: -5},
  max_pooling2d_1: { type:'pooling',filterSize:2,stride:1, heading: '2x2 pooling, 1x1 strides', scalingFactor: 2, z: 45 },
  dropout_1: { type:'dropout', heading: 'p=0.25 (only active during training phase)', scalingFactor: 2 },
  flatten_1: { type:'flatten', heading: '', scalingFactor: 2 , z: 95},
  dense_1: { type:'fc', outDim: 128, heading: 'output dimensions 128', scalingFactor: 4, z: 135 },
  activation_3: { type:'activation', heading: 'ReLU', scalingFactor: 4, z: 185 },
  dropout_2: { type:'dropout', heading: 'p=0.5 (only active during training phase)', scalingFactor: 4 },
  dense_2: { type:'fc', outDim: 128, heading: 'output dimensions 10', scalingFactor: 8 , z: 235 },
  activation_4: { type:'softmax', heading: 'Softmax', scalingFactor: 8 , z: 285}
}

export default {
  props: ['hasWebgl'],

  data: function() {
    return {
      useGpu: this.hasWebgl,
      model: new KerasJS.Model(Object.assign({ gpu: this.hasWebgl }, MODEL_CONFIG)), // eslint-disable-line
      modelLoading: true,
      input: new Float32Array(784),
      output: new Float32Array(10),
      outputClasses: range(10),
      layerResultImages: [],
      layerDisplayConfig: LAYER_DISPLAY_CONFIG,
      drawing: false,
      strokes: []
    }
  },

  watch: {
    useGpu: function(value) {
      this.model.toggleGpu(value)
    }
  },

  computed: {
    loadingProgress: function() {
      return this.model.getLoadingProgress()
    },
    predictedClass: function() {
      if (this.output.reduce((a, b) => a + b, 0) === 0) {
        return -1
      }
      return this.output.reduce((argmax, n, i) => (n > this.output[argmax] ? i : argmax), 0)
    }
  },

  mounted: function() {
    this.model.ready().then(() => {
      this.modelLoading = false
      this.$nextTick(() => {
        this.initWebgl()
        this.getIntermediateResults()
      })
    })
  },

  methods: {
	initWebgl: function() {			
		const container = document.getElementById("webgl_container");
        this.container = container;

        this.originalWidth = 0.95*container.offsetWidth ;
		this.originalHeight = 1000;
        this.rotatingCam = false;
		this.posX = [];
		this.posY = [];
		this.posZ = [];
		this.layerNum = [];
		this.points = null;
		this.pickedPoint = null;
		this.lines = null;
		this.clicked = false;

		const scene = new THREE.Scene();
        //scene.background = new THREE.Color( 0xff0000 );
        this.scene = scene;

		scene.fog = new THREE.Fog( 0xffffff, 1, 10000 );
		scene.fog.color.setHSL( 0.6, 0, 1 );

		const highlightBox = new THREE.Mesh(
			new THREE.BoxGeometry( 12,12,12 ),
			new THREE.MeshLambertMaterial( { color: 0xffff00 }
		) );
		highlightBox.visible = false;
		scene.add( highlightBox );
        this.highlightBox = highlightBox;

	    const camera = new THREE.PerspectiveCamera( 75, this.originalWidth / this.originalHeight, 1, 15000 );
	    camera.position.z = 5000;
        this.camera = camera;

		const pickingScene = new THREE.Scene();
		const pickingTexture = new THREE.WebGLRenderTarget( this.originalWidth, this.originalHeight);
		pickingTexture.minFilter = THREE.LinearFilter;
		pickingTexture.generateMipmaps = false;
        this.pickingScene = pickingScene;
        this.pickingTexture = pickingTexture;

	    const renderer = new THREE.WebGLRenderer({ antialias: true});
        renderer.setClearColor( 0xFFAAFF );
		renderer.setPixelRatio( window.devicePixelRatio );
		renderer.sortObjects = false;
	    renderer.setSize( this.originalWidth, this.originalHeight );
        this.renderer = renderer;

		var cameraControls = new THREE.OrbitControls( this.camera, renderer.domElement );
		cameraControls.target.set( 0, 0, 0 );

		var loader = new THREE.FontLoader();
		loader.load( 'https://raw.githubusercontent.com/rollup/three-jsnext/master/examples/fonts/droid/droid_sans_bold.typeface.json',this.setFont);

		this.raycaster = new THREE.Raycaster();
		this.mouse = new THREE.Vector2();

	    container.appendChild( renderer.domElement );
		renderer.domElement.addEventListener( 'mousemove', this.onMouseMove );
		renderer.domElement.addEventListener( 'mousedown', this.onMouseDown );
		renderer.domElement.addEventListener( 'click', this.onClick );
		renderer.domElement.addEventListener( 'mouseup', this.onMouseUp );
		window.addEventListener( 'resize', this.onWindowResize, false );
		this.animate();
	},

	animate: function() {
		requestAnimationFrame( this.animate );
		this.render();
    },
 
	render: function() {
		if(!this.clicked){
			this.raycaster.setFromCamera( this.mouse, this.camera );
			if(this.points !== null){
				var intersects = this.raycaster.intersectObject( this.points );
				if ( intersects.length > 0 ) {
					console.log(intersects);
					if ( this.pickedPoint != intersects[ 0 ].index ) {
						//attributes.size.array[ this.pickedPoint ] = 14;
						//this.pickedPoint = intersects[ 0 ].index;
						//attributes.size.array[ this.pickedPoint ] = 16;
						//attributes.size.needsUpdate = true;
					}
				} else if ( this.pickedPoint !== null ) {
					//attributes.size.array[ this.pickedPoint ] = 14;
					//attributes.size.needsUpdate = true;
					//this.pickedPoint = null;
				}
			}
		}
		this.renderer.render( this.scene, this.camera );
	},

	drawEdges: function() {
		var lineMat = new THREE.LineBasicMaterial({
			color: 0x0000ff,
			transparent:true, 
			linewidth: 2
		});
		var lineGeom = new THREE.Geometry();
		lineGeom.dynamic = true;
		var line = new THREE.Line(lineGeom, lineMat);
		line.name = 'edges';
		this.lines = line;
		scene.add(line);
	},

	updateEdges: function() {
		var r=1, g=1, b=1, rw=1, gw, bw, i, j, v, colorNum, weight;
		var vertAdjust = 3;
		var numChildren = scene.children.length;
		var colors = [];
		vertCount = 0;
		for ( var c = 0; c<numChildren; c++) {
			if ( scene.children[c].name == 'edges' ){
				var object = scene.children[c];
				object.geometry.dispose();

				var lineGeom = new THREE.Geometry();
				lineGeom.dynamic = true;						
								
				lineGeom.vertices.push(new THREE.Vector3(posX[ind_below], posY[ind_below]+vertAdjust, posZ[ind_below]));
				lineGeom.vertices.push(new THREE.Vector3(posX[ind], posY[ind]-3, posZ[ind]));
								
				colors[ vertCount ] = new THREE.Color( rw,gw,bw );
				vertCount++;
				colors[ vertCount ] = new THREE.Color( rw,gw,bw );
				vertCount++;
			}
		}
		var lineMat = new THREE.LineBasicMaterial( { color: 0xffffff, opacity: 1, linewidth: 1, vertexColors: THREE.VertexColors } );						
		object.material = lineMat;
		object.geometry.colors = colors;
		object.geometry.vertices = lineGeom.vertices;
		object.material.needsUpdate = true;
		object.geometry.colorsNeedUpdate = true;
		object.geometry.verticesNeedUpdate = true;
	},

	setFont: function( font ) {
		this.font = font;
	},
		
	onWindowResize: function( e ) {
			//render();
	},
		
	onMouseDown: function( e ) {
		this.rotatingCam = true;
		this.clicked = true;
	},
	
	onClick: function( e ) {
		//console.log('click');
	},
	
	onMouseUp: function( e ) {
		//console.log('mouse up');
		this.rotatingCam = false;	
		this.clicked = false;
		
	},

	onMouseMove: function( e ) {
		event.preventDefault();
		this.mouse.x = ( e.clientX / window.innerWidth ) * 2 - 1;
		this.mouse.y = - ( e.clientY / window.innerHeight ) * 2 + 1;
	},

    clear: function() {
      const ctx = document.getElementById('input-canvas').getContext('2d')
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
      const ctxCenterCrop = document.getElementById('input-canvas-centercrop').getContext('2d')
      ctxCenterCrop.clearRect(0, 0, ctxCenterCrop.canvas.width, ctxCenterCrop.canvas.height)
      const ctxScaled = document.getElementById('input-canvas-scaled').getContext('2d')
      ctxScaled.clearRect(0, 0, ctxScaled.canvas.width, ctxScaled.canvas.height)
      this.output = new Float32Array(10)
      this.drawing = false
      this.strokes = []
	  while(this.scene.children.length > 0) { 
        this.scene.remove(this.scene.children[0]); 
	  }
    },
    activateDraw: function(e) {
      this.drawing = true
      this.strokes.push([])
      let points = this.strokes[this.strokes.length - 1]
      points.push(utils.getCoordinates(e))
    },
    draw: function(e) {
      if (!this.drawing) return

      const ctx = document.getElementById('input-canvas').getContext('2d')

      ctx.lineWidth = 20
      ctx.lineJoin = ctx.lineCap = 'round'
      ctx.strokeStyle = '#393E46'

      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)

      let points = this.strokes[this.strokes.length - 1]
      points.push(utils.getCoordinates(e))

      // draw individual strokes
      for (let s = 0, slen = this.strokes.length; s < slen; s++) {
        points = this.strokes[s]

        let p1 = points[0]
        let p2 = points[1]
        ctx.beginPath()
        ctx.moveTo(...p1)

        // draw points in stroke
        // quadratic bezier curve
        for (let i = 1, len = points.length; i < len; i++) {
          ctx.quadraticCurveTo(...p1, ...utils.getMidpoint(p1, p2))
          p1 = points[i]
          p2 = points[i + 1]
        }
        ctx.lineTo(...p1)
        ctx.stroke()
      }
    },
    deactivateDrawAndPredict: debounce(
      function() {
        if (!this.drawing) return
        this.drawing = false

        const ctx = document.getElementById('input-canvas').getContext('2d')

        // center crop
        const imageDataCenterCrop = utils.centerCrop(ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height))
        const ctxCenterCrop = document.getElementById('input-canvas-centercrop').getContext('2d')
        ctxCenterCrop.canvas.width = imageDataCenterCrop.width
        ctxCenterCrop.canvas.height = imageDataCenterCrop.height
        ctxCenterCrop.putImageData(imageDataCenterCrop, 0, 0)

        // scaled to 28 x 28
        const ctxScaled = document.getElementById('input-canvas-scaled').getContext('2d')
        ctxScaled.save()
        ctxScaled.scale(28 / ctxCenterCrop.canvas.width, 28 / ctxCenterCrop.canvas.height)
        ctxScaled.clearRect(0, 0, ctxCenterCrop.canvas.width, ctxCenterCrop.canvas.height)
        ctxScaled.drawImage(document.getElementById('input-canvas-centercrop'), 0, 0)
        const imageDataScaled = ctxScaled.getImageData(0, 0, ctxScaled.canvas.width, ctxScaled.canvas.height)
        ctxScaled.restore()

        // process image data for model input
        const { data } = imageDataScaled
        this.input = new Float32Array(784)
        for (let i = 0, len = data.length; i < len; i += 4) {
          this.input[i / 4] = data[i + 3] / 255
        }

        this.model.predict({ input: this.input }).then(outputData => {
          this.output = outputData.output;

		  while(this.scene.children.length > 0) { 
		    this.scene.remove(this.scene.children[0]); 
		  }

          this.getIntermediateResults();
        })
      },
      200,
      { leading: true, trailing: true }
    ),
    getIntermediateResults: function() {
      let results = []
      for (let [name, layer] of this.model.modelLayersMap.entries()) {
        if (name.startsWith('dropout')) continue

        const layerClass = layer.layerClass || ''

        let images = []
		console.log(layer)
		if (layer.result) {
		    if (layer.result.tensor.shape.length === 3) {
		      images = utils.unroll3Dtensor(layer.result.tensor)
		    } else if (layer.result.tensor.shape.length === 2) {
		      images = [utils.image2Dtensor(layer.result.tensor)]
		    } else if (layer.result.tensor.shape.length === 1) {
		      images = [utils.image1Dtensor(layer.result.tensor)]
		    }
		    results.push({ name, layerClass, images })
		}
      }
      this.layerResultImages = results
      setTimeout(() => {
        this.displayOutput()
      }, 0)
    },
	displayOutput: function() {
        if(this.layerResultImages.length <= 1) {
			return;	
		}
		//if(this.points === null) { 
		var geometry = new THREE.Geometry();
		var colors = [];
		geometry.colors = colors;

		// material
		var material = new THREE.PointsMaterial( {
			size: 14,
			transparent: false,
			opacity: 0.9,
			vertexColors: THREE.VertexColors
		} );

		this.pointLayers = [];
		this.pointImages = [];
		this.pointMap = {};
		var nPoints = 0;
	    var layerX = 0; var layerY = 0; var layerZ = 0; var height = 0; var width = 0; var fc = false; var xPos = 0; var yPos = 0; var spacing = 0; var len = 0;
		this.layerResultImages.forEach((result, layerNum) => {
	        var images = result.images;
	    	const scalingFactor = this.layerDisplayConfig[result.name].scalingFactor;
	    	const zPos = this.layerDisplayConfig[result.name].z;
	        len = Math.ceil(Math.sqrt(images.length));
	        layerZ = -layerNum * 450;
	    	layerY =  0;
	        if(images.length==1 && images[0].height==1) {
	            fc = true;
	            spacing = 18;
			} else {
	            fc = false;
	            spacing = 10;
	    		xPos = layerY - len/2 *images[0].width*11;
				yPos = layerX + len/2 *images[0].height*11;
				var text = this.createLabel(result.name, 0, yPos+100, layerZ, 100, "white");
				this.scene.add(text);
			}
	    	images.forEach((image, imageNum) => {
	            width = image.width;
	            height = image.height;
	            if(fc) {  
	                len = Math.ceil(Math.sqrt(width));
	            	width = len;
	            	height = Math.ceil(width/height);
	        		xPos = layerX - len/2 * 18;
	    			yPos = layerY + len/2 * 18;
					var text = this.createLabel(result.name, 0, yPos+100, layerZ, 30, "white");
					this.scene.add(text);
	            } else {
	        		xPos = layerY + (-len/2 + imageNum%len)*image.width*11;
	    			yPos = layerX - (-len/2 + Math.floor(imageNum/len))*image.height*11;
	            }
				for ( var xInd = 0; xInd < width; xInd ++ ) {
					for ( var yInd = 0; yInd < height; yInd ++ ) {
			            if(fc && (xInd*width + yInd >= image.width)) { 
			            	break;
			            }
						var position = new THREE.Vector3();
						position.y = yPos-xInd*spacing;
						position.x = xPos+yInd*spacing;
						position.z = layerZ;

						this.pointMap[position] = nPoints;
						this.pointLayers.push(layerNum);
						this.pointImages.push(imageNum);
						// add it to the geometry
						geometry.vertices.push(position);
		
						var v = image.data[(xInd*width+yInd)*4+3];
	                    colors.push(new THREE.Color( v/255.0,v/255.0,v/255.0));
	                    nPoints++;
					}
				}
			});
		});
		var pointCloud = new THREE.Points( geometry, material );
		this.scene.add( pointCloud );
		this.points = pointCloud;
		/*} else {
			var nPoints = 0;
			this.layerResultImages.forEach((result, layerNum) => {
		    	result.images.forEach((image, imageNum) => {
					for ( var xInd = 0; xInd < image.width; xInd ++ ) {
						for ( var yInd = 0; yInd < image.height; yInd ++ ) {
							var v = image.data[(xInd*width+yInd)*4+3];
		                    this.points.geometry.colors[nPoints].setRGB(v/255.0,v/255.0,v/255.0);
							nPoints++;
						}
					}
				});
			});
			this.points.geometry.colorsNeedUpdate = true;
		}*/		
	},
    createLabel: function(text, x, y, z, size, color) {
		var materialFront = new THREE.MeshBasicMaterial( { color: color} );
		var materialSide = new THREE.MeshBasicMaterial( { color: 0x000088 } );
		var materialArray = [ materialFront, materialSide ];
		var textGeom = new THREE.TextGeometry( text, 
		{
			size: size, height: 4, curveSegments: 3, font: this.font,
			bevelThickness: 1, bevelSize: 2, bevelEnabled: true,
			material: 0, extrudeMaterial: 1
		});
	
		var textMaterial = new THREE.MeshFaceMaterial(materialArray);
		var textMesh = new THREE.Mesh(textGeom, textMaterial );
		textGeom.computeBoundingBox();
    	textGeom.textWidth = textGeom.boundingBox.max.x - textGeom.boundingBox.min.x;
		textMesh.position.set( x-textGeom.textWidth/2, y, z );
		return textMesh;
	}
  }
}
</script>

<style scoped>
@import '../../variables.css';

.demo.mnist-cnn {
  & .column {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  & .column.input-column {
    justify-content: flex-end;

    & .input-container {
      text-align: right;
      margin: 20px;
      position: relative;
      user-select: none;

      & .input-label {
        font-family: var(--font-cursive);
        font-size: 18px;
        color: var(--color-lightgray);
        text-align: right;

        & span.arrow {
          font-size: 36px;
          color: #CCCCCC;
          position: absolute;
          right: -32px;
          top: 8px;
        }
      }

      & .canvas-container {
        display: inline-flex;
        justify-content: flex-end;
        margin: 10px 0;
        border: 15px solid var(--color-green-lighter);
        transition: border-color 0.2s ease-in;

        &:hover {
          border-color: var(--color-green-light);
        }

        & canvas {
          background: whitesmoke;

          &:hover {
            cursor: crosshair;
          }
        }
      }

      & .input-buttons {
        display: flex;
        align-items: center;
        justify-content: flex-end;

        & .input-clear-button {
          display: flex;
          align-items: center;
          color: var(--color-lightgray);
          transition: color 0.2s ease-in;

          & .material-icons {
            margin-right: 5px;
          }

          &:hover {
            color: var(--color-green-lighter);
            cursor: pointer;
          }
        }
      }
    }
  }

  & .column.controls-column {
    align-items: flex-start;
    justify-content: flex-start;
    padding-top: 80px;
  }

  & .column.output-column {
    justify-content: center;

    & .output {
      height: 160px;
      display: flex;
      flex-direction: row;
      align-items: flex-end;
      justify-content: center;
      user-select: none;
      cursor: default;

      & .output-class {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 0 6px;
        border-bottom: 2px solid var(--color-green-lighter);

        & .output-label {
          font-family: var(--font-monospace);
          font-size: 1.5rem;
          color: var(--color-lightgray);
        }

        & .output-bar {
          width: 8px;
          background: #EEEEEE;
          transition: height 0.2s ease-out;
        }
      }

      & .output-class.predicted {
        border-bottom-color: var(--color-green);

        & .output-label {
          color: var(--color-green);
        }
      }
    }
  }

  & .layer-results-container {
    position: relative;
    height: 1000px;
    & #webgl_container {
		z-index: 10;
        display: block;
    }

    & .bg-line {
      position: absolute;
      z-index: 0;
      top: 0;
      left: 50%;
      background: whitesmoke;
      width: 15px;
      height: 100%;
    }

    & .layer-result {
      position: relative;
      z-index: 1;
      margin: 30px 20px;
      background: whitesmoke;
      border-radius: 10px;
      padding: 20px;
      overflow-x: auto;

      & .layer-result-heading {
        font-size: 1rem;
        color: #999999;
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
        font-size: 12px;

        & span.layer-class {
          color: var(--color-green);
          font-size: 14px;
          font-weight: bold;
        }
      }

      & .layer-result-canvas-container {
        display: inline-flex;
        flex-wrap: wrap;
        background: whitesmoke;

        & canvas {
          border: 1px solid lightgray;
          margin: 1px;
        }
      }
    }
  }
}

</style>
