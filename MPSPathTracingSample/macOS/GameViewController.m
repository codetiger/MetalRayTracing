/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Implementation for our macOS view controller
*/

#import "GameViewController.h"
#import "Renderer.h"

@implementation GameViewController
{
    MTKView *_view;

    Renderer *_renderer;
}

- (void)viewDidLoad
{
    [super viewDidLoad];

    _view = (MTKView *)self.view;
    
    // Set color space of view to SRGB  
    CGColorSpaceRef colorSpace = CGColorSpaceCreateWithName(kCGColorSpaceLinearSRGB);
    _view.colorspace = colorSpace;
    CGColorSpaceRelease(colorSpace);
    
    // Lookup high power GPU if this is a discrete GPU system
    NSArray<id <MTLDevice>> *devices = MTLCopyAllDevices();
    
    id <MTLDevice> device = devices[0];
    
    for (id <MTLDevice> potentialDevice in devices) {
        if (!potentialDevice.lowPower) {
            device = potentialDevice;
            break;
        }
    }

    _view.device = device;

    if(!_view.device)
    {
        NSLog(@"Metal is not supported on this device");
        self.view = [[NSView alloc] initWithFrame:self.view.frame];
        return;
    }

    _renderer = [[Renderer alloc] initWithMetalKitView:_view];

    [_renderer mtkView:_view drawableSizeWillChange:_view.bounds.size];

    _view.delegate = _renderer;
}

@end
